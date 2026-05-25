#!/usr/bin/env python3
"""
Compute custom linear weights from Statcast RE24 data.

Includes standard batting events plus:
  - Strikeouts (separated from other outs)
  - Stolen bases and caught stealing
  - Pitches per PA (marginal value per pitch)

Methodology:
  1. Build a 24-state run expectancy (RE) matrix from PA-level data
  2. Compute RE24 for each PA: RE(state_after) + runs_scored - RE(state_before)
  3. Average RE24 by event type → raw linear weights
  4. Derive SB/CS run values from RE matrix state transitions
  5. Regression: RE24 ~ event_dummies + pitch_count → pitch value

Usage:
    python compute_weights.py
    python compute_weights.py --years 2023,2024,2025
"""

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pybaseball


# ── Event classification ─────────────────────────────────────────────
PA_EVENTS = {
    "single", "double", "triple", "home_run",
    "walk", "intent_walk", "hit_by_pitch",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play",
    "fielders_choice", "fielders_choice_out",
    "field_error", "sac_fly", "sac_bunt",
    "sac_fly_double_play", "sac_bunt_double_play",
    "catcher_interf", "other_out",
}

K_EVENTS = {"strikeout", "strikeout_double_play"}

SB_EVENTS = {
    "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
}
CS_EVENTS = {
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
}

POSITIVE_EVENTS = {"single", "double", "triple", "home_run",
                   "walk", "intent_walk", "hit_by_pitch"}

WOBA_DENOM_EXCLUDE = {"intent_walk", "catcher_interf"}

EVENT_GROUP = {}
for e in PA_EVENTS:
    if e in {"single"}:           EVENT_GROUP[e] = "1B"
    elif e in {"double"}:         EVENT_GROUP[e] = "2B"
    elif e in {"triple"}:         EVENT_GROUP[e] = "3B"
    elif e in {"home_run"}:       EVENT_GROUP[e] = "HR"
    elif e in {"walk", "intent_walk"}: EVENT_GROUP[e] = "BB"
    elif e in {"hit_by_pitch"}:   EVENT_GROUP[e] = "HBP"
    elif e in K_EVENTS:           EVENT_GROUP[e] = "K"
    else:                         EVENT_GROUP[e] = "Other Out"


# ── FanGraphs reference weights (for comparison) ─────────────────────
FANGRAPHS_WEIGHTS = {
    2023: {"BB": 0.696, "HBP": 0.726, "1B": 0.883, "2B": 1.244, "3B": 1.569, "HR": 2.015},
    2024: {"BB": 0.689, "HBP": 0.720, "1B": 0.882, "2B": 1.254, "3B": 1.590, "HR": 2.050},
    2025: {"BB": 0.691, "HBP": 0.722, "1B": 0.882, "2B": 1.252, "3B": 1.584, "HR": 2.037},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute linear weights from Statcast RE24")
    parser.add_argument("--years", help="Comma-separated years (default: current + 2 prior)")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════
def fetch_statcast(years):
    pybaseball.cache.enable()
    frames = []
    for year in years:
        start = f"{year}-03-15"
        end_dt = min(date.today(), date(year, 11, 15))
        end = end_dt.strftime("%Y-%m-%d")
        print(f"    {year}: {start} → {end} …")
        df = pybaseball.statcast(start_dt=start, end_dt=end)
        if df.empty:
            print(f"    No data for {year}.")
            continue
        if "game_type" in df.columns:
            df = df[df["game_type"] == "R"]
        df["year"] = year
        frames.append(df)
        print(f"      {len(df):,} pitch records")
    if not frames:
        print("  No data returned. Exiting.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# PA DATASET
# ══════════════════════════════════════════════════════════════════════
def build_pa_dataset(df):
    """
    Build one row per PA from pitch-level data.
    Uses the first pitch of each at-bat for the initial base-out state,
    and the last pitch (with events) for the outcome.
    """
    required = ["game_pk", "at_bat_number", "inning", "inning_topbot",
                "events", "outs_when_up", "on_1b", "on_2b", "on_3b",
                "bat_score", "post_bat_score", "pitch_number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Missing columns: {missing}")
        sys.exit(1)

    df_sorted = df.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    # First pitch of each at-bat → initial state
    first = df_sorted.groupby(["game_pk", "at_bat_number"]).first().reset_index()
    first = first[["game_pk", "at_bat_number", "inning", "inning_topbot",
                    "on_1b", "on_2b", "on_3b", "outs_when_up", "bat_score"]].copy()
    first.columns = ["game_pk", "ab", "inning", "topbot",
                     "r1", "r2", "r3", "outs", "score_before"]
    first["r1"] = first["r1"].notna().astype(int)
    first["r2"] = first["r2"].notna().astype(int)
    first["r3"] = first["r3"].notna().astype(int)

    # Last pitch with an event → outcome
    events_df = df_sorted[df_sorted["events"].notna()].copy()
    last = events_df.groupby(["game_pk", "at_bat_number"]).last().reset_index()
    last = last[["game_pk", "at_bat_number", "events",
                 "post_bat_score", "pitch_number"]].copy()
    last.columns = ["game_pk", "ab", "events", "score_after", "pitches"]

    pa = first.merge(last, on=["game_pk", "ab"], how="inner")
    pa["runs_on_play"] = pa["score_after"] - pa["score_before"]

    # Half-inning identifier
    pa["hi"] = (pa["game_pk"].astype(str) + "_"
                + pa["inning"].astype(str) + "_"
                + pa["topbot"])

    # Filter to real PA events
    pa_mask = pa["events"].isin(PA_EVENTS)
    pa_only = pa[pa_mask].copy()

    # Base-out state string: "RRR_O" (e.g., "100_1" = runner on 1st, 1 out)
    pa_only["state"] = (pa_only["r1"].astype(str)
                        + pa_only["r2"].astype(str)
                        + pa_only["r3"].astype(str)
                        + "_" + pa_only["outs"].astype(str))

    # Event group
    pa_only["group"] = pa_only["events"].map(EVENT_GROUP)

    print(f"    {len(pa_only):,} PAs")
    return pa_only


def extract_sb_cs(df):
    """
    Extract SB/CS events from ALL pitch rows.
    SB/CS often happen mid-PA on individual pitches rather than as
    standalone events, so we scan the description field.
    """
    if "des" not in df.columns and "description" not in df.columns:
        print("    No description column found; skipping SB/CS extraction.")
        return pd.DataFrame()

    desc_col = "des" if "des" in df.columns else "description"
    df_desc = df[df[desc_col].notna()].copy()

    sb_mask = df_desc[desc_col].str.contains(
        r"steals|stolen", case=False, na=False
    )
    cs_mask = df_desc[desc_col].str.contains(
        r"caught stealing|picked off", case=False, na=False
    )
    combined = df_desc[sb_mask | cs_mask].copy()

    if combined.empty:
        print("    No SB/CS events found in descriptions.")
        return pd.DataFrame()

    combined["r1"] = combined["on_1b"].notna().astype(int)
    combined["r2"] = combined["on_2b"].notna().astype(int)
    combined["r3"] = combined["on_3b"].notna().astype(int)
    combined["outs"] = combined["outs_when_up"].astype(int)
    combined["runs_on_play"] = combined["post_bat_score"] - combined["bat_score"]

    combined["sb_cs_type"] = "unknown"
    combined.loc[sb_mask, "sb_cs_type"] = "SB"
    combined.loc[cs_mask, "sb_cs_type"] = "CS"

    desc_lower = combined[desc_col].str.lower()
    combined["target_base"] = "2b"
    combined.loc[desc_lower.str.contains("3rd|third|steals 3|to 3", na=False), "target_base"] = "3b"
    combined.loc[desc_lower.str.contains("home|scores", na=False), "target_base"] = "home"

    print(f"    {sb_mask.sum():,} SB events  |  {cs_mask.sum():,} CS events")
    return combined


# ══════════════════════════════════════════════════════════════════════
# RUN EXPECTANCY MATRIX
# ══════════════════════════════════════════════════════════════════════
def compute_re_matrix(pa):
    """
    Compute the 24-state run expectancy matrix.
    RE(state) = average runs scored from that state to end of half-inning.
    """
    # Total runs per half-inning
    hi_agg = pa.groupby("hi").agg(
        hi_start=("score_before", "min"),
        hi_end=("score_after", "max"),
    )
    hi_agg["hi_runs"] = hi_agg["hi_end"] - hi_agg["hi_start"]
    pa = pa.merge(hi_agg[["hi_runs", "hi_start"]], left_on="hi", right_index=True)

    # Runs remaining from each PA to end of half-inning
    pa["runs_already"] = pa["score_before"] - pa["hi_start"]
    pa["runs_remaining"] = pa["hi_runs"] - pa["runs_already"]

    re = pa.groupby("state").agg(
        re=("runs_remaining", "mean"),
        count=("runs_remaining", "size"),
    )
    return re


def print_re_matrix(re):
    print(f"\n  {'='*54}")
    print(f"  {'24-State Run Expectancy Matrix':^54}")
    print(f"  {'='*54}")

    base_labels = {
        "000": "Empty", "100": "1__", "010": "_2_", "001": "__3",
        "110": "12_",  "101": "1_3", "011": "_23", "111": "123",
    }
    print(f"\n  {'Runners':<10} {'0 out':>10} {'1 out':>10} {'2 out':>10}")
    print(f"  {'─'*44}")
    for bases in ["000", "100", "010", "001", "110", "101", "011", "111"]:
        vals = []
        for outs in [0, 1, 2]:
            state = f"{bases}_{outs}"
            if state in re.index:
                vals.append(f"{re.loc[state, 're']:.3f}")
            else:
                vals.append("  N/A")
        label = base_labels.get(bases, bases)
        print(f"  {label:<10} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")
    print()


# ══════════════════════════════════════════════════════════════════════
# RE24 PER PA
# ══════════════════════════════════════════════════════════════════════
def compute_re24(pa, re_matrix):
    """
    RE24 = RE(state_after) + runs_on_play - RE(state_before)
    state_after is determined from the NEXT PA in the same half-inning.
    If this is the last PA of the half-inning, RE(state_after) = 0.
    """
    pa = pa.sort_values(["hi", "ab"]).reset_index(drop=True)

    # RE before
    pa["re_before"] = pa["state"].map(re_matrix["re"])

    # Next PA's state in the same half-inning
    pa["next_state"] = pa.groupby("hi")["state"].shift(-1)
    pa["re_after"] = pa["next_state"].map(re_matrix["re"]).fillna(0.0)

    pa["re24"] = pa["re_after"] + pa["runs_on_play"] - pa["re_before"]

    return pa


# ══════════════════════════════════════════════════════════════════════
# LINEAR WEIGHTS BY EVENT TYPE
# ══════════════════════════════════════════════════════════════════════
def compute_event_weights(pa):
    """Average RE24 per event group → raw linear weights."""
    grouped = pa.groupby("group").agg(
        re24_mean=("re24", "mean"),
        re24_std=("re24", "std"),
        count=("re24", "size"),
    ).sort_values("re24_mean", ascending=False)
    return grouped


def print_event_weights(weights, years):
    print(f"\n  {'='*70}")
    print(f"  {'Raw Linear Weights (RE24 per event)':^70}")
    print(f"  {'='*70}")

    # Get average out value for computing wOBA-scale weights
    out_re24 = 0.0
    out_count = 0
    for grp in ["K", "Other Out"]:
        if grp in weights.index:
            out_re24 += weights.loc[grp, "re24_mean"] * weights.loc[grp, "count"]
            out_count += weights.loc[grp, "count"]
    avg_out_re24 = out_re24 / out_count if out_count > 0 else 0

    print(f"\n  {'Event':<12} {'RE24':>8} {'Count':>10} {'Std':>8}"
          f"  {'Shifted':>8}  {'FG wt':>8}")
    print(f"  {'─'*62}")

    fg = {}
    for yr in years:
        if yr in FANGRAPHS_WEIGHTS:
            for k, v in FANGRAPHS_WEIGHTS[yr].items():
                fg[k] = v

    display_order = ["HR", "3B", "2B", "1B", "HBP", "BB", "K", "Other Out"]
    for grp in display_order:
        if grp not in weights.index:
            continue
        row = weights.loc[grp]
        shifted = row["re24_mean"] - avg_out_re24
        fg_val = fg.get(grp, None)
        fg_str = f"{fg_val:.3f}" if fg_val else "  —"
        print(f"  {grp:<12} {row['re24_mean']:>+8.3f} {int(row['count']):>10,}"
              f" {row['re24_std']:>8.3f}  {shifted:>8.3f}  {fg_str:>8}")

    print(f"\n  Avg out RE24: {avg_out_re24:+.4f}")
    k_re24 = weights.loc["K", "re24_mean"] if "K" in weights.index else 0
    other_out_re24 = weights.loc["Other Out", "re24_mean"] if "Other Out" in weights.index else 0
    print(f"  K vs Other Out:  {k_re24 - other_out_re24:+.4f} runs"
          f"  (K is {'worse' if k_re24 < other_out_re24 else 'better'})")
    print()


# ══════════════════════════════════════════════════════════════════════
# SB / CS VALUES FROM RE MATRIX
# ══════════════════════════════════════════════════════════════════════
def compute_sb_cs_values(sb_cs_df, re_matrix):
    """
    Compute average RE24 for SB and CS events using the RE matrix
    to derive state transitions.
    """
    results = {}

    if sb_cs_df.empty:
        return {"SB": {"re24": 0.0, "count": 0, "std": 0.0},
                "CS": {"re24": 0.0, "count": 0, "std": 0.0}}

    for label in ["SB", "CS"]:
        subset = sb_cs_df[sb_cs_df["sb_cs_type"] == label]
        if subset.empty:
            results[label] = {"re24": 0.0, "count": 0, "std": 0.0}
            continue

        is_sb = label == "SB"
        re24_vals = []
        for _, row in subset.iterrows():
            r1, r2, r3 = int(row["r1"]), int(row["r2"]), int(row["r3"])
            outs = int(row["outs"])
            state_before = f"{r1}{r2}{r3}_{outs}"
            re_before = re_matrix.loc[state_before, "re"] if state_before in re_matrix.index else 0
            runs = row["runs_on_play"]
            target = row["target_base"]

            if is_sb:
                if target == "2b":
                    nr1, nr2, nr3 = 0, 1, r3
                elif target == "3b":
                    nr1, nr2, nr3 = r1, 0, 1
                elif target == "home":
                    nr1, nr2, nr3 = r1, r2, 0
                else:
                    continue
                n_outs = outs
            else:
                if target == "2b":
                    nr1, nr2, nr3 = 0, r2, r3
                elif target == "3b":
                    nr1, nr2, nr3 = r1, 0, r3
                elif target == "home":
                    nr1, nr2, nr3 = r1, r2, 0
                else:
                    continue
                n_outs = outs + 1

            if n_outs >= 3:
                re_after = 0.0
            else:
                state_after = f"{nr1}{nr2}{nr3}_{n_outs}"
                re_after = re_matrix.loc[state_after, "re"] if state_after in re_matrix.index else 0

            re24_vals.append(re_after + runs - re_before)

        results[label] = {
            "re24": float(np.mean(re24_vals)) if re24_vals else 0.0,
            "count": len(re24_vals),
            "std": float(np.std(re24_vals)) if re24_vals else 0.0,
        }

    return results


def print_sb_cs(results):
    print(f"\n  {'='*50}")
    print(f"  {'Stolen Base / Caught Stealing Run Values':^50}")
    print(f"  {'='*50}")
    print(f"\n  {'Event':<6} {'RE24':>8} {'Count':>8} {'Std':>8}")
    print(f"  {'─'*34}")
    for label in ["SB", "CS"]:
        r = results[label]
        print(f"  {label:<6} {r['re24']:>+8.3f} {r['count']:>8,} {r.get('std', 0):>8.3f}")

    sb_re = results["SB"]["re24"]
    cs_re = results["CS"]["re24"]
    if sb_re != 0 and cs_re != 0:
        breakeven = -cs_re / (sb_re - cs_re)
        print(f"\n  Break-even SB success rate: {breakeven:.1%}")
        print(f"  (Need >{breakeven:.1%} success rate for SB to add value)")
    print()


# ══════════════════════════════════════════════════════════════════════
# REGRESSION: RE24 ~ EVENT TYPE + PITCH COUNT
# ══════════════════════════════════════════════════════════════════════
def run_pitch_regression(pa):
    """
    OLS: RE24 = β₀ + Σ(β_group × I(group)) + β_pitch × pitches + ε
    Baseline = Other Out (omitted dummy).
    """
    groups = ["1B", "2B", "3B", "HR", "BB", "HBP", "K"]
    X_cols = []
    for g in groups:
        pa[f"is_{g}"] = (pa["group"] == g).astype(float)
        X_cols.append(f"is_{g}")

    pa["pitches_centered"] = pa["pitches"] - pa["pitches"].mean()
    X_cols.append("pitches_centered")

    X = pa[X_cols].values.astype(np.float64)
    X = np.column_stack([np.ones(len(X)), X])
    y = pa["re24"].values.astype(np.float64)

    # OLS via normal equations
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    # Residuals and standard errors
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape
    s2 = (resid @ resid) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))

    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))

    col_names = ["Intercept"] + groups + ["Pitch (centered)"]
    results = {
        "coefficients": dict(zip(col_names, beta)),
        "std_errors": dict(zip(col_names, se)),
        "r_squared": r2,
        "n": n,
        "avg_pitches": float(pa["pitches"].mean()),
    }
    return results


def print_regression(reg):
    print(f"\n  {'='*64}")
    print(f"  {'Regression: RE24 ~ Event Type + Pitch Count':^64}")
    print(f"  {'='*64}")
    print(f"\n  N = {reg['n']:,}   R² = {reg['r_squared']:.4f}"
          f"   Avg pitches/PA = {reg['avg_pitches']:.2f}")

    print(f"\n  {'Variable':<18} {'Coeff':>10} {'Std Err':>10} {'t-stat':>10}")
    print(f"  {'─'*52}")

    display_order = ["Intercept", "HR", "3B", "2B", "1B", "HBP", "BB", "K", "Pitch (centered)"]
    for var in display_order:
        if var not in reg["coefficients"]:
            continue
        coef = reg["coefficients"][var]
        se = reg["std_errors"][var]
        t = coef / se if se > 0 else 0
        sig = " ***" if abs(t) > 3.29 else " **" if abs(t) > 2.58 else " *" if abs(t) > 1.96 else ""
        print(f"  {var:<18} {coef:>+10.4f} {se:>10.4f} {t:>10.2f}{sig}")

    pitch_coef = reg["coefficients"].get("Pitch (centered)", 0)
    print(f"\n  Interpretation:")
    print(f"    Intercept = avg RE24 of Other Out (baseline)")
    print(f"    Event coefficients = RE24 relative to Other Out")
    print(f"    Pitch coefficient = {pitch_coef:+.4f} runs per additional pitch")
    print(f"    → A 7-pitch PA is worth ~{pitch_coef * (7 - reg['avg_pitches']):+.3f} runs")
    print(f"       vs avg ({reg['avg_pitches']:.1f} pitches), controlling for outcome")
    print()


# ══════════════════════════════════════════════════════════════════════
# CAVEATS
# ══════════════════════════════════════════════════════════════════════
def print_caveats():
    print(f"\n  {'='*70}")
    print(f"  {'CAVEATS':^70}")
    print(f"  {'='*70}")
    caveats = [
        ("Strikeouts vs Other Outs",
         "The difference is real but small (~0.01-0.03 runs). Strikeouts\n"
         "    cannot advance runners or produce sacrifice flies. Including K\n"
         "    as a separate weight is methodologically sound."),

        ("Stolen Bases / Caught Stealing",
         "SB and CS are baserunning events, not batting events. Including\n"
         "    them in a batter's wRC+ conflates hitting with baserunning.\n"
         "    FanGraphs intentionally excludes them from wOBA/wRC+. If you\n"
         "    include them, the metric becomes more like 'total offensive\n"
         "    contribution' rather than pure batting value."),

        ("Pitches Per PA",
         "The marginal pitch value is small and partially endogenous:\n"
         "    high pitch counts correlate with walks (positive) and also with\n"
         "    at-bats where the hitter is struggling (foul balls before K).\n"
         "    The regression coefficient captures value CONDITIONAL on the\n"
         "    outcome, meaning it reflects pitcher fatigue / pitch quality\n"
         "    degradation, not the full value of patience. Double-counting\n"
         "    risk exists since patient hitters already benefit from more BB."),

        ("SB/CS Attribution",
         "SB/CS events in Statcast may occur mid-PA. Our RE24 computation\n"
         "    uses the base-out state at the first pitch of each PA, so mid-PA\n"
         "    SB/CS are captured in the PA outcome's context rather than as\n"
         "    separate events. The separate SB/CS values are computed from\n"
         "    events that end at-bats (mostly CS with 2 outs)."),

        ("Regression R²",
         "R² is typically low (~0.10-0.15) because individual PA outcomes\n"
         "    are inherently noisy. This is expected and does not indicate a\n"
         "    poor model — it reflects the variance of baseball."),
    ]
    for i, (title, text) in enumerate(caveats, 1):
        print(f"\n  {i}. {title}")
        print(f"    {text}")
    print()


# ══════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════
def export_weights(event_weights, sb_cs, reg, years):
    """Export computed weights to JSON."""
    out_re24 = 0.0
    out_count = 0
    for grp in ["K", "Other Out"]:
        if grp in event_weights.index:
            out_re24 += event_weights.loc[grp, "re24_mean"] * event_weights.loc[grp, "count"]
            out_count += event_weights.loc[grp, "count"]
    avg_out = out_re24 / out_count if out_count > 0 else 0

    weights = {}
    for grp in event_weights.index:
        weights[grp] = {
            "re24": round(float(event_weights.loc[grp, "re24_mean"]), 4),
            "shifted": round(float(event_weights.loc[grp, "re24_mean"] - avg_out), 4),
            "count": int(event_weights.loc[grp, "count"]),
        }

    output = {
        "years": years,
        "event_weights": weights,
        "avg_out_re24": round(avg_out, 4),
        "sb_cs": {
            "SB": round(sb_cs["SB"]["re24"], 4),
            "CS": round(sb_cs["CS"]["re24"], 4),
        },
        "pitch_value": round(reg["coefficients"].get("Pitch (centered)", 0), 5),
        "avg_pitches_per_pa": round(reg["avg_pitches"], 2),
        "regression_r2": round(reg["r_squared"], 4),
    }

    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "custom_weights.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Exported to {path}\n")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    if args.years:
        years = sorted(int(y) for y in args.years.split(","))
    else:
        current = date.today().year
        years = [current - 2, current - 1, current]

    print(f"\n{'='*70}")
    print(f"  Custom Linear Weights — RE24 Regression")
    print(f"  Years: {', '.join(map(str, years))}")
    print(f"{'='*70}\n")

    # 1. Fetch data
    print("── Fetching Statcast data ──────────────────────────────────\n")
    df = fetch_statcast(years)

    # 2. Build PA dataset
    print("\n── Building PA data