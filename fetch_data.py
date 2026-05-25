#!/usr/bin/env python3
"""
Fetch multi-year Statcast plate appearance data and produce JSON for the
visualization.

Usage:
    python fetch_data.py                  # current year + 2 prior
    python fetch_data.py --years 2025     # single year
    python fetch_data.py --years 2024,2025,2026

After running, start a local server to view:
    python -m http.server 8000
    open http://localhost:8000
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime

import numpy as np
import pandas as pd
import pybaseball


# ── wOBA linear weights + league constants (FanGraphs Guts!) ─────────
# Updated yearly — https://www.fangraphs.com/tools/guts
# wK derived from RE24 regression: K is ~0.019 runs worse than avg out,
# converted to wOBA scale: wK = -0.019 × wOBAScale
WEIGHTS_BY_YEAR = {
    2023: {
        "wBB": 0.696, "wHBP": 0.726, "w1B": 0.883,
        "w2B": 1.244, "w3B": 1.569, "wHR": 2.015,
        "wK": -0.023,
        "lgwOBA": 0.318, "wOBAScale": 1.204, "lgRPA": 0.119,
    },
    2024: {
        "wBB": 0.689, "wHBP": 0.720, "w1B": 0.882,
        "w2B": 1.254, "w3B": 1.590, "wHR": 2.050,
        "wK": -0.024,
        "lgwOBA": 0.310, "wOBAScale": 1.242, "lgRPA": 0.117,
    },
    2025: {
        "wBB": 0.691, "wHBP": 0.722, "w1B": 0.882,
        "w2B": 1.252, "w3B": 1.584, "wHR": 2.037,
        "wK": -0.023,
        "lgwOBA": 0.313, "wOBAScale": 1.232, "lgRPA": 0.118,
    },
}

EVENT_WEIGHT_KEY = {
    "single":                "w1B",
    "double":                "w2B",
    "triple":                "w3B",
    "home_run":              "wHR",
    "walk":                  "wBB",
    "intent_walk":           "wBB",
    "hit_by_pitch":          "wHBP",
    "strikeout":             "wK",
    "strikeout_double_play": "wK",
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

GDP_EVENTS = {"grounded_into_double_play", "double_play", "triple_play"}

# RE24-derived run values from compute_weights.py
SB_RUN_VALUE = 0.2
CS_RUN_VALUE = -0.5
GDP_EXTRA_COST = 0.39


def get_weights(year: int) -> dict:
    if year in WEIGHTS_BY_YEAR:
        return WEIGHTS_BY_YEAR[year]
    fallback = max(WEIGHTS_BY_YEAR.keys())
    print(f"    No weights for {year}; using {fallback} as proxy.")
    return WEIGHTS_BY_YEAR[fallback]


def run_value(event: str, weights: dict) -> float:
    key = EVENT_WEIGHT_KEY.get(event)
    return weights[key] if key else 0.0


def fetch_season(year: int) -> pd.DataFrame:
    """Fetch one season of Statcast data, regular-season games only."""
    pybaseball.cache.enable()
    start = f"{year}-03-15"
    end = min(date.today(), date(year, 11, 15)).strftime("%Y-%m-%d")

    print(f"  {year}: fetching {start} → {end} …")
    df = pybaseball.statcast(start_dt=start, end_dt=end)
    if df.empty:
        return df

    before = len(df)
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    excluded = before - len(df)
    if excluded:
        print(f"    Excluded {excluded:,} spring-training / exhibition rows")
    print(f"    {len(df):,} regular-season pitch records")
    return df


def compute_bsr(full_df: pd.DataFrame, all_pa: pd.DataFrame) -> dict:
    """
    Compute simplified BsR (Baserunning Runs) per batter from Statcast.
    Components:
      wSB  – weighted stolen base runs (SB/CS attributed to runners)
      wGDP – weighted GDP avoidance runs
    Note: UBR (extra bases on hits/outs) is omitted.
    """
    desc_col = "des" if "des" in full_df.columns else "description"
    if desc_col not in full_df.columns:
        print("    No description column; skipping BsR.")
        return {}

    # ── wSB: attribute SB/CS to the runner ─────────────────────────
    desc_lower = full_df[desc_col].fillna("").str.lower()
    sb_mask = desc_lower.str.contains("steals", na=False) & ~desc_lower.str.contains("caught", na=False)
    cs_mask = desc_lower.str.contains("caught stealing", na=False)

    runner_events: dict[int, dict] = defaultdict(lambda: {"sb": 0, "cs": 0})

    for mask_series, evt_type in [(sb_mask, "sb"), (cs_mask, "cs")]:
        subset = full_df[mask_series]
        if subset.empty:
            continue
        sub_desc = subset[desc_col].str.lower()
        to_2nd = sub_desc.str.contains("2nd|second", na=False, regex=True)
        to_3rd = sub_desc.str.contains("3rd|third", na=False, regex=True)
        to_home = sub_desc.str.contains("home|scores", na=False, regex=True)

        for cond, col in [(to_2nd, "on_1b"), (to_3rd, "on_2b"), (to_home, "on_3b")]:
            ids = subset.loc[cond, col].dropna()
            for rid, count in ids.value_counts().items():
                runner_events[int(rid)][evt_type] += int(count)

    total_sb = sum(v["sb"] for v in runner_events.values())
    total_cs = sum(v["cs"] for v in runner_events.values())
    total_pa = len(all_pa)
    lg_wsb_per_pa = (total_sb * SB_RUN_VALUE + total_cs * CS_RUN_VALUE) / total_pa if total_pa else 0

    print(f"    SB/CS: {total_sb:,} SB, {total_cs:,} CS across all batters")

    # ── wGDP: GDP avoidance vs league rate ─────────────────────────
    gdp_opp = all_pa["on_1b"].notna() & (all_pa["outs_when_up"] < 2)
    is_gdp = all_pa["events"].isin(GDP_EVENTS)
    lg_gdp_rate = is_gdp[gdp_opp].mean() if gdp_opp.any() else 0

    print(f"    GDP: league rate = {lg_gdp_rate:.3f} in {gdp_opp.sum():,} opportunities")

    # ── per-batter aggregation ─────────────────────────────────────
    bsr_dict: dict[int, dict] = {}
    pa_grouped = all_pa.groupby("batter")

    for bid, group in pa_grouped:
        bid = int(bid)
        pa_count = len(group)

        sb = runner_events[bid]["sb"]
        cs = runner_events[bid]["cs"]
        wsb = (sb * SB_RUN_VALUE + cs * CS_RUN_VALUE) - (lg_wsb_per_pa * pa_count)

        opps = (group["on_1b"].notna() & (group["outs_when_up"] < 2)).sum()
        gdps = group["events"].isin(GDP_EVENTS).sum()
        if opps > 0:
            player_gdp_rate = gdps / opps
            wgdp = (lg_gdp_rate - player_gdp_rate) * opps * GDP_EXTRA_COST
        else:
            wgdp = 0.0

        bsr_dict[bid] = {
            "bsr": round(float(wsb + wgdp), 2),
            "sb": int(sb),
            "cs": int(cs),
        }

    return bsr_dict


def build_player_json(all_pa: pd.DataFrame) -> tuple[list[dict], dict[str, int]]:
    """Returns (player_list, name_to_id_map)."""
    batter_ids = all_pa["batter"].unique().tolist()
    print(f"  Looking up names for {len(batter_ids):,} unique batters …")
    name_df = pybaseball.playerid_reverse_lookup(batter_ids, ke