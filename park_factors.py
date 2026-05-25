#!/usr/bin/env python3
"""
Park Factor Exploration Script

Computes park factors from Statcast data with configurable parameters,
tests reliability via split-half correlations, and compares against
FanGraphs published values.

Usage:
    python park_factors.py
"""

import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pybaseball


# ══════════════════════════════════════════════════════════════════════
# CONFIGURABLE PARAMETERS — adjust these and re-run
# ══════════════════════════════════════════════════════════════════════
YEARS = range(2018, 2027)
REGRESSION_GAMES = 0        # 0 = no regression; try 60-80 for moderate
SPLIT_BY_HANDEDNESS = True
MIN_PA_PER_PARK = 500
EXCLUDE_HOME_BATTERS = True  # only use visiting-team PAs at each park


# ══════════════════════════════════════════════════════════════════════
# RENOVATION / NEW STADIUM CUTOFFS
# Data before this year is excluded for the given team's home park.
# ══════════════════════════════════════════════════════════════════════
RENOVATION_CUTOFFS = {
    "TEX": 2020,  # Globe Life Field opened
    "ATL": 2017,  # Truist Park opened
    "TOR": 2023,  # Rogers Centre walls moved in / raised
    "BAL": 2022,  # "Walltimore" left-field wall changes
    "LAA": 2018,  # HR line lowered 10 inches
    "KC":  2026,  # Kauffman walls moved in
    "COL": 2023,  # Humidor extended to bullpens
    "OAK": 2025,  # Sutter Health Park (temporary minor-league park)
}


# ══════════════════════════════════════════════════════════════════════
# wOBA LINEAR WEIGHTS BY YEAR  (FanGraphs Guts!)
# ══════════════════════════════════════════════════════════════════════
WEIGHTS_BY_YEAR = {
    2018: {"wBB": 0.690, "wHBP": 0.720, "w1B": 0.880, "w2B": 1.247,
           "w3B": 1.578, "wHR": 2.031,
           "lgwOBA": 0.315, "wOBAScale": 1.226, "lgRPA": 0.120},
    2019: {"wBB": 0.690, "wHBP": 0.720, "w1B": 0.870, "w2B": 1.243,
           "w3B": 1.559, "wHR": 2.015,
           "lgwOBA": 0.320, "wOBAScale": 1.157, "lgRPA": 0.126},
    2020: {"wBB": 0.699, "wHBP": 0.730, "w1B": 0.886, "w2B": 1.264,
           "w3B": 1.594, "wHR": 2.042,
           "lgwOBA": 0.319, "wOBAScale": 1.221, "lgRPA": 0.121},
    2021: {"wBB": 0.692, "wHBP": 0.722, "w1B": 0.879, "w2B": 1.242,
           "w3B": 1.568, "wHR": 2.015,
           "lgwOBA": 0.313, "wOBAScale": 1.224, "lgRPA": 0.117},
    2022: {"wBB": 0.689, "wHBP": 0.720, "w1B": 0.881, "w2B": 1.248,
           "w3B": 1.576, "wHR": 2.032,
           "lgwOBA": 0.310, "wOBAScale": 1.244, "lgRPA": 0.116},
    2023: {"wBB": 0.696, "wHBP": 0.726, "w1B": 0.883, "w2B": 1.244,
           "w3B": 1.569, "wHR": 2.015,
           "lgwOBA": 0.318, "wOBAScale": 1.204, "lgRPA": 0.119},
    2024: {"wBB": 0.689, "wHBP": 0.720, "w1B": 0.882, "w2B": 1.254,
           "w3B": 1.590, "wHR": 2.050,
           "lgwOBA": 0.310, "wOBAScale": 1.242, "lgRPA": 0.117},
    2025: {"wBB": 0.691, "wHBP": 0.722, "w1B": 0.882, "w2B": 1.252,
           "w3B": 1.584, "wHR": 2.037,
           "lgwOBA": 0.313, "wOBAScale": 1.232, "lgRPA": 0.118},
}

EVENT_WEIGHT_KEY = {
    "single": "w1B", "double": "w2B", "triple": "w3B", "home_run": "wHR",
    "walk": "wBB", "intent_walk": "wBB", "hit_by_pitch": "wHBP",
}

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

# FanGraphs published park factors (Basic, ~2023-2025 3-yr avg, 100=neutral)
# Source: https://www.fangraphs.com/guts.aspx?type=pf
# These are approximate — the exact values shift as FanGraphs updates.
FANGRAPHS_PF = {
    "ARI": 103, "ATL": 100, "BAL": 98,  "BOS": 110, "CHC": 102,
    "CIN": 106, "CLE": 98,  "COL": 125, "CWS": 102, "DET": 102,
    "HOU": 103, "KC":  100, "LAA": 97,  "LAD": 96,  "MIA": 98,
    "MIL": 99,  "MIN": 101, "NYM": 97,  "NYY": 103, "OAK": 96,
    "PHI": 102, "PIT": 98,  "SD":  94,  "SF":  97,  "SEA": 96,
    "STL": 97,  "TB":  92,  "TEX": 94,  "TOR": 99,  "WSH": 100,
}

WOBA_DENOM_EXCLUDE = {
    "intent_walk", "sac_bunt", "sac_bunt_double_play", "catcher_interf",
}


def get_weights(year: int) -> dict:
    if year in WEIGHTS_BY_YEAR:
        return WEIGHTS_BY_YEAR[year]
    return WEIGHTS_BY_YEAR[max(WEIGHTS_BY_YEAR.keys())]


def run_value(event: str, weights: dict) -> float:
    key = EVENT_WEIGHT_KEY.get(event)
    return weights[key] if key else 0.0


# ══════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════
def fetch_all_pa(years) -> pd.DataFrame:
    """Fetch Statcast PA data for all requested years."""
    pybaseball.cache.enable()
    frames = []
    for year in years:
        start = f"{year}-03-15"
        end = min(date.today(), date(year, 11, 15)).strftime("%Y-%m-%d")
        print(f"  {year}: fetching {start} → {end} …")
        df = pybaseball.statcast(start_dt=start, end_dt=end)
        if df.empty:
            print(f"    No data for {year}, skipping.")
            continue

        if "game_type" in df.columns:
            before = len(df)
            df = df[df["game_type"] == "R"]
            excl = before - len(df)
            if excl:
                print(f"    Excluded {excl:,} non-regular-season rows")

        pa = df[df["events"].isin(PA_EVENTS)].copy()
        weights = get_weights(year)
        pa["run_value"] = pa["events"].apply(lambda e: run_value(e, weights))
        pa["year"] = year

        keep_cols = [
            "batter", "stand", "home_team", "away_team",
            "inning_topbot", "events", "run_value",
            "estimated_woba_using_speedangle", "year",
        ]
        frames.append(pa[[c for c in keep_cols if c in pa.columns]])
        print(f"    {len(pa):,} PAs")

    if not frames:
        print("  No data returned for any year. Exiting.")
        sys.exit(1)

    all_pa = pd.concat(frames, ignore_index=True)
    print(f"\n  Total: {len(all_pa):,} PAs across {len(frames)} season(s)\n")
    return all_pa


# ══════════════════════════════════════════════════════════════════════
# PARK FACTOR COMPUTATION
# ══════════════════════════════════════════════════════════════════════
def compute_park_factors(
    pa: pd.DataFrame,
    split_hand: bool = True,
    regression_games: int = 0,
    min_pa: int = 500,
    renovation_cutoffs: dict | None = None,
    year_filter: list[int] | None = None,
    exclude_home_batters: bool = False,
) -> pd.DataFrame:
    """
    For each stadium, compare mean run_value of PAs at that stadium vs.
    mean run_value of all other PAs league-wide (excluding that stadium).
    Computed per-year then averaged, so year-to-year league trends cancel out.

    If exclude_home_batters=True, only visiting-team PAs are used at each
    park, removing the bias from teams building rosters for their home park.
    """
    if renovation_cutoffs is None:
        renovation_cutoffs = {}

    if year_filter is not None:
        pa = pa[pa["year"].isin(year_filter)]

    results = []
    teams = sorted(pa["home_team"].unique())
    all_years = sorted(pa["year"].unique())

    for team in teams:
        cutoff = renovation_cutoffs.get(team, int(pa["year"].min()))
        valid_years = [y for y in all_years if y >= cutoff]
        if not valid_years:
            continue

        hands = ["L", "R"] if split_hand else [None]
        for hand in hands:
            yearly_ratios = []
            total_home_pa = 0
            total_rest_pa = 0

            for yr in valid_years:
                yr_pa = pa[pa["year"] == yr]
                at_park = yr_pa[yr_pa["home_team"] == team]
                if exclude_home_batters:
                    at_park = at_park[at_park["inning_topbot"] == "Top"]
                rest = yr_pa[yr_pa["home_team"] != team]

                if hand:
                    at_park = at_park[at_park["stand"] == hand]
                    rest = rest[rest["stand"] == hand]

                if len(at_park) < 50 or len(rest) < 50:
                    continue

                park_mean = at_park["run_value"].mean()
                rest_mean = rest["run_value"].mean()
                if rest_mean == 0:
                    continue

                yearly_ratios.append({
                    "ratio": park_mean / rest_mean,
                    "park_pa": len(at_park),
                    "rest_pa": len(rest),
                    "park_rv": park_mean,
                    "rest_rv": rest_mean,
                })
                total_home_pa += len(at_park)
                total_rest_pa += len(rest)

            if not yearly_ratios or total_home_pa < min_pa:
                continue

            weights = [r["park_pa"] for r in yearly_ratios]
            raw_pf = np.average(
                [r["ratio"] for r in yearly_ratios], weights=weights
            )

            if regression_games > 0:
                approx_games = total_home_pa / 38
                pf = (raw_pf * approx_games + 1.0 * regression_games) / (
                    approx_games + regression_games
                )
            else:
                pf = raw_pf

            avg_park_rv = np.average(
                [r["park_rv"] for r in yearly_ratios], weights=weights
            )
            avg_rest_rv = np.average(
                [r["rest_rv"] for r in yearly_ratios], weights=weights
            )

            results.append({
                "team": team,
                "hand": hand or "All",
                "pf": round(pf * 100, 1),
                "pf_raw": round(raw_pf * 100, 1),
                "home_pa": total_home_pa,
                "rest_pa": total_rest_pa,
                "home_rv": round(avg_park_rv, 4),
                "rest_rv": round(avg_rest_rv, 4),
                "years": f"{min(valid_years)}-{max(valid_years)}" if len(valid_years) > 1 else str(valid_years[0]),
                "n_years": len(valid_years),
            })

    if split_hand:
        all_pf = compute_park_factors(
            pa, split_hand=False, regression_games=regression_games,
            min_pa=min_pa, renovation_cutoffs=renovation_cutoffs,
            year_filter=year_filter, exclude_home_batters=exclude_home_batters,
        )
        return pd.concat([all_pf, pd.DataFrame(results)], ignore_index=True)

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════
# SPLIT-HALF CORRELATION
# ══════════════════════════════════════════════════════════════════════
def split_half_correlation(pa: pd.DataFrame, renovation_cutoffs: dict) -> dict:
    """
    Split years into odd/even, compute PFs independently, correlate.
    """
    all_years = sorted(pa["year"].unique())
    odd_years = [y for y in all_years if y % 2 == 1]
    even_years = [y for y in all_years if y % 2 == 0]

    print(f"  Odd years:  {odd_years}")
    print(f"  Even years: {even_years}")

    pf_odd = compute_park_factors(
        pa, split_hand=SPLIT_BY_HANDEDNESS, regression_games=REGRESSION_GAMES,
        min_pa=max(200, MIN_PA_PER_PARK // 2), renovation_cutoffs=renovation_cutoffs,
        year_filter=odd_years, exclude_home_batters=EXCLUDE_HOME_BATTERS,
    )
    pf_even = compute_park_factors(
        pa, split_hand=SPLIT_BY_HANDEDNESS, regression_games=REGRESSION_GAMES,
        min_pa=max(200, MIN_PA_PER_PARK // 2), renovation_cutoffs=renovation_cutoffs,
        year_filter=even_years, exclude_home_batters=EXCLUDE_HOME_BATTERS,
    )

    correlations = {}
    for hand_label in (["All", "L", "R"] if SPLIT_BY_HANDEDNESS else ["All"]):
        odd_h = pf_odd[pf_odd["hand"] == hand_label].set_index("team")["pf"]
        even_h = pf_even[pf_even["hand"] == hand_label].set_index("team")["pf"]
        common = odd_h.index.intersection(even_h.index)
        if len(common) < 5:
            correlations[hand_label] = (np.nan, len(common))
            continue
        r = np.corrcoef(odd_h[common].values, even_h[common].values)[0, 1]
        correlations[hand_label] = (round(r, 3), len(common))

    return correlations


# ══════════════════════════════════════════════════════════════════════
# PER-GAME ADJUSTMENT PREVIEW
# ══════════════════════════════════════════════════════════════════════
def per_game_preview(pa: pd.DataFrame, pf_df: pd.DataFrame):
    """
    For a few high-PA players, show wRC+ with and without per-game
    park adjustment to illustrate the impact.
    """
    pf_all = pf_df[pf_df["hand"] == "All"].set_index("team")["pf"]
    pf_l = pf_df[pf_df["hand"] == "L"].set_index("team")["pf"] if SPLIT_BY_HANDEDNESS else None
    pf_r = pf_df[pf_df["hand"] == "R"].set_index("team")["pf"] if SPLIT_BY_HANDEDNESS else None

    recent = pa[pa["year"] >= pa["year"].max()]
    batter_pa_counts = recent.groupby("batter").size()
    top_batters = batter_pa_counts.nlargest(15).index

    rows = []
    for bid in top_batters:
        bpa = recent[recent["batter"] == bid]
        n_pa = len(bpa)
        if n_pa < 200:
            continue

        in_denom = ~bpa["events"].isin(WOBA_DENOM_EXCLUDE)
        woba_n = bpa["run_value"].sum()
        woba_d = in_denom.sum()
        if woba_d == 0:
            continue

        year = int(bpa["year"].iloc[0])
        w = get_weights(year)
        woba = woba_n / woba_d
        wraa_pa = (woba - w["lgwOBA"]) / w["wOBAScale"]
        wrc_plus_nopf = round(((wraa_pa + w["lgRPA"]) / w["lgRPA"]) * 100)

        adj_rv_sum = 0.0
        adj_denom = 0
        for _, row in bpa.iterrows():
            stadium = row["home_team"]
            hand = row.get("stand")
            if SPLIT_BY_HANDEDNESS and pf_l is not None and pf_r is not None:
                if hand == "L" and stadium in pf_l.index:
                    pf_val = pf_l[stadium] / 100
                elif hand == "R" and stadium in pf_r.index:
                    pf_val = pf_r[stadium] / 100
                elif stadium in pf_all.index:
                    pf_val = pf_all[stadium] / 100
                else:
                    pf_val = 1.0
            elif stadium in pf_all.index:
                pf_val = pf_all[stadium] / 100
            else:
                pf_val = 1.0

            adj_rv_sum += row["run_value"]
            if row["events"] not in WOBA_DENOM_EXCLUDE:
                adj_denom += 1

        if adj_denom == 0:
            continue

        home_team_mode = bpa.apply(
            lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"],
            axis=1,
        ).mode()
        player_team = home_team_mode.iloc[0] if len(home_team_mode) > 0 else "???"

        adj_woba = adj_rv_sum / adj_denom
        pf_adjustments = []
        for _, row in bpa.iterrows():
            stadium = row["home_team"]
            hand = row.get("stand")
            if SPLIT_BY_HANDEDNESS and pf_l is not None and pf_r is not None:
                if hand == "L" and stadium in pf_l.index:
                    pf_val = pf_l[stadium] / 100
                elif hand == "R" and stadium in pf_r.index:
                    pf_val = pf_r[stadium] / 100
                elif stadium in pf_all.index:
                    pf_val = pf_all[stadium] / 100
                else:
                    pf_val = 1.0
            elif stadium in pf_all.index:
                pf_val = pf_all[stadium] / 100
            else:
                pf_val = 1.0
            pf_adjustments.append(pf_val)

        mean_pf = np.mean(pf_adjustments)
        wraa_pa_adj = (woba - w["lgwOBA"]) / w["wOBAScale"]
        wrc_plus_adj = round(
            ((wraa_pa_adj + w["lgRPA"] + (w["lgRPA"] - mean_pf * w["lgRPA"]))
             / w["lgRPA"]) * 100
        )

        rows.append({
            "batter_id": bid,
            "team": player_team,
            "pa": n_pa,
            "wRC+ (no PF)": wrc_plus_nopf,
            "wRC+ (per-game PF)": wrc_plus_adj,
            "diff": wrc_plus_adj - wrc_plus_nopf,
            "avg_PF": round(mean_pf * 100, 1),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# PRINTING HELPERS
# ══════════════════════════════════════════════════════════════════════
def print_header(title: str):
    w = 70
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}\n")


def print_park_factors(pf_df: pd.DataFrame):
    print_header("PARK FACTORS")
    print(f"  Config: regression={REGRESSION_GAMES} games, "
          f"min_PA={MIN_PA_PER_PARK}, "
          f"handedness={'yes' if SPLIT_BY_HANDEDNESS else 'no'}\n")

    for hand_label in (["All"] + (["L", "R"] if SPLIT_BY_HANDEDNESS else [])):
        subset = pf_df[pf_df["hand"] == hand_label].sort_values("pf", ascending=False)
        label = {"All": "OVERALL", "L": "LEFT-HANDED BATTERS", "R": "RIGHT-HANDED BATTERS"}[hand_label]
        print(f"  ── {label} {'─' * (55 - len(label))}")
        print(f"  {'Team':<6} {'PF':>6} {'Raw':>6} {'ParkPAs':>8} {'RestPAs':>9} {'ParkRV':>8} {'RestRV':>8} {'Years':>10}")
        print(f"  {'─' * 69}")
        for _, r in subset.iterrows():
            print(f"  {r['team']:<6} {r['pf']:>6.1f} {r['pf_raw']:>6.1f} "
                  f"{r['home_pa']:>8,} {r['rest_pa']:>9,} "
                  f"{r['home_rv']:>8.4f} {r['rest_rv']:>8.4f} {r['years']:>10}")
        print()


def print_split_half(correlations: dict):
    print_header("SPLIT-HALF RELIABILITY (odd vs even years)")
    print(f"  {'Split':<8} {'r':>8} {'N teams':>10}  Interpretation")
    print(f"  {'─' * 50}")
    for hand, (r, n) in correlations.items():
        if np.isnan(r):
            interp = "insufficient data"
        elif r >= 0.8:
            interp = "strong — factors are stable"
        elif r >= 0.6:
            interp = "moderate — usable with caution"
        elif r >= 0.4:
            interp = "weak — consider more years or regression"
        else:
            interp = "poor — too noisy to rely on"
        label = {"All": "Overall", "L": "LHB", "R": "RHB"}.get(hand, hand)
        print(f"  {label:<8} {r:>8.3f} {n:>10}  {interp}")
    print()


def print_fangraphs_comparison(pf_df: pd.DataFrame):
    print_header("FANGRAPHS COMPARISON")
    pf_all = pf_df[pf_df["hand"] == "All"].set_index("team")["pf"]
    common = sorted(set(pf_all.index) & set(FANGRAPHS_PF.keys()))

    if not common:
        print("  No common teams to compare.")
        return

    print(f"  {'Team':<6} {'Computed':>10} {'FanGraphs':>10} {'Diff':>8} {'Flag':>6}")
    print(f"  {'─' * 46}")
    diffs = []
    for team in common:
        comp = pf_all[team]
        fg = FANGRAPHS_PF[team]
        diff = comp - fg
        diffs.append(diff)
        flag = " ***" if abs(diff) > 5 else ""
        print(f"  {team:<6} {comp:>10.1f} {fg:>10} {diff:>+8.1f}{flag}")

    computed_vals = [pf_all[t] for t in common]
    fg_vals = [FANGRAPHS_PF[t] for t in common]
    r = np.corrcoef(computed_vals, fg_vals)[0, 1]
    mad = np.mean(np.abs(diffs))

    print(f"\n  Correlation with FanGraphs:    r = {r:.3f}")
    print(f"  Mean absolute difference:     {mad:.1f} points")
    print(f"  Parks with >5pt difference:   {sum(1 for d in diffs if abs(d) > 5)} / {len(common)}")
    print(f"\n  Note: FanGraphs uses 3-5yr rolling avg WITH regression and")
    print(f"  without handedness. Differences are expected.\n")


def print_per_game_preview(preview: pd.DataFrame):
    print_header("PER-GAME PARK ADJUSTMENT PREVIEW (most recent season)")
    if preview.empty:
        print("  No players with enough PAs.\n")
        return

    preview = preview.sort_values("diff", key=abs, ascending=False)
    print(f"  {'Team':<6} {'PA':>5} {'wRC+(noPF)':>11} {'wRC+(pgPF)':>11} {'Diff':>6} {'AvgPF':>7}")
    print(f"  {'─' * 52}")
    for _, r in preview.iterrows():
        print(f"  {r['team']:<6} {r['pa']:>5} {r['wRC+ (no PF)']:>11} "
              f"{r['wRC+ (per-game PF)']:>11} {r['diff']:>+6} {r['avg_PF']:>7.1f}")
    print(f"\n  Positive diff = player was in pitcher-friendly parks more often")
    print(f"  Negative diff = player was in hitter-friendly parks more often\n")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'=' * 70}")
    print(f"  Park Factor Exploration")
    print(f"  Years: {min(YEARS)}-{max(YEARS)} | "
          f"Regression: {REGRESSION_GAMES} games | "
          f"Handedness: {'on' if SPLIT_BY_HANDEDNESS else 'off'} | "
          f"Home batters: {'excluded' if EXCLUDE_HOME_BATTERS else 'included'}")
    print(f"{'=' * 70}\n")

    # 1. Fetch data
    print("─── Fetching Statcast data ───────────────────────────────\n")
    all_pa = fetch_all_pa(YEARS)

    # 2. Compute park factors
    print("─── Computing park factors ──────────────────────────────\n")
    pf = compute_park_factors(
        all_pa,
        split_hand=SPLIT_BY_HANDEDNESS,
        regression_games=REGRESSION_GAMES,
        min_pa=MIN_PA_PER_PARK,
        renovation_cutoffs=RENOVATION_CUTOFFS,
        exclude_home_batters=EXCLUDE_HOME_BATTERS,
    )
    print_park_factors(pf)

    # 3. Split-half reliability
    print("─── Running split-half test ─────────────────────────────\n")
    correlations = split_half_correlation(all_pa, RENOVATION_CUTOFFS)
    print_split_half(correlations)

    # 4. FanGraphs comparison
    print_fangraphs_comparison(pf)

    # 5. Per-game adjustment preview
    print("─── Per-game adjustment preview ─────────────────────────\n")
    preview = per_game_preview(all_pa, pf)
    print_per_game_preview(preview)

    # 6. Export to JSON for the website
    export_json(pf)


def export_json(pf_df: pd.DataFrame):
    """Export park factors to data/park_factors.json for use by the website."""
    out = {}
    for hand_label in ["All", "L", "R"]:
        subset = pf_df[pf_df["hand"] == hand_label]
        if subset.empty:
            continue
        out[hand_label] = {
            row["team"]: row["pf"] for _, row in subset.iterrows()
        }

    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "park_factors.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print_header("EXPORTED")
    print(f"  Wrote {path}")
    print(f"  Keys: {list(out.keys())}")
    print(f"  Teams per key: {', '.join(str(len(v)) for v in out.values())}\n")


if __name__ == "__main__":
    main()
