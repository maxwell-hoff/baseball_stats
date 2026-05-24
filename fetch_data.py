#!/usr/bin/env python3
"""
Fetch Statcast plate appearance data and produce JSON for the visualization.

Usage:
    python fetch_data.py              # fetch current season
    python fetch_data.py --year 2025  # fetch a specific season

After running, start a local server to view:
    python -m http.server 8000
    open http://localhost:8000
"""

import argparse
import json
import os
import sys
from datetime import date, datetime

import pandas as pd
import pybaseball


# ── wOBA linear weights (FanGraphs Guts!) ────────────────────────────
# These represent the run-value contribution of each batting event.
# Updated yearly — swap in new values from https://www.fangraphs.com/tools/guts
WEIGHTS_BY_YEAR = {
    2024: {
        "wBB": 0.689, "wHBP": 0.720, "w1B": 0.882,
        "w2B": 1.254, "w3B": 1.590, "wHR": 2.050,
        "lgwOBA": 0.310, "wOBAScale": 1.242, "lgRPA": 0.117,
    },
    2025: {
        "wBB": 0.691, "wHBP": 0.722, "w1B": 0.882,
        "w2B": 1.252, "w3B": 1.584, "wHR": 2.037,
        "lgwOBA": 0.313, "wOBAScale": 1.232, "lgRPA": 0.118,
    },
}

# Statcast event → linear-weight mapping key
EVENT_WEIGHT_KEY = {
    "single":       "w1B",
    "double":       "w2B",
    "triple":       "w3B",
    "home_run":     "wHR",
    "walk":         "wBB",
    "intent_walk":  "wBB",
    "hit_by_pitch": "wHBP",
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "intent_walk"}

# Events excluded from the wOBA denominator (AB + BB - IBB + SF + HBP)
WOBA_DENOM_EXCLUDE = {"intent_walk", "sac_bunt", "sac_bunt_double_play", "catcher_interf"}

# Events that end a plate appearance (excludes baserunning events like
# caught_stealing, pickoff, etc. that also appear in the events column)
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


def get_weights(year: int) -> dict:
    """Return the weight dict for *year*, falling back to the most recent."""
    if year in WEIGHTS_BY_YEAR:
        return WEIGHTS_BY_YEAR[year]
    fallback = max(WEIGHTS_BY_YEAR.keys())
    print(f"  No weights for {year}; using {fallback} weights as proxy.")
    return WEIGHTS_BY_YEAR[fallback]


def run_value(event: str, weights: dict) -> float:
    key = EVENT_WEIGHT_KEY.get(event)
    return weights[key] if key else 0.0


def fetch_statcast(year: int) -> pd.DataFrame:
    """Download Statcast data for the given season."""
    pybaseball.cache.enable()

    start = f"{year}-03-20"
    end = min(date.today(), date(year, 11, 15)).strftime("%Y-%m-%d")

    print(f"  Fetching Statcast data  {start} → {end} …")
    print("  (this can take a few minutes on the first run; results are cached)")
    df = pybaseball.statcast(start_dt=start, end_dt=end)
    print(f"  Received {len(df):,} pitch records.")
    return df


def build_player_json(df: pd.DataFrame, weights: dict) -> list[dict]:
    """Aggregate per-player stats and individual PA event lists."""
    pa = df[df["events"].isin(PA_EVENTS)].copy()
    pa["run_value"] = pa["events"].apply(lambda e: run_value(e, weights))
    pa["game_date_str"] = pd.to_datetime(pa["game_date"]).dt.strftime("%Y-%m-%d")

    # Determine each batter's team from the most-recent game
    pa["batting_team"] = pa.apply(
        lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"],
        axis=1,
    )

    # player_name column is the *pitcher*, not the batter.
    # Look up batter names from their MLBAM IDs.
    batter_ids = pa["batter"].unique().tolist()
    print(f"  Looking up names for {len(batter_ids):,} unique batters …")
    name_df = pybaseball.playerid_reverse_lookup(batter_ids, key_type="mlbam")
    batter_names: dict[int, str] = {}
    for _, row in name_df.iterrows():
        first = str(row["name_first"]).strip().title()
        last = str(row["name_last"]).strip().title()
        batter_names[int(row["key_mlbam"])] = f"{first} {last}"

    players: dict[int, dict] = {}
    for row in pa.itertuples(index=False):
        bid = int(row.batter)
        if bid not in players:
            players[bid] = {
                "name": batter_names.get(bid, f"Unknown ({bid})"),
                "team": "",
                "pa": 0,
                "hits": 0,
                "walks": 0,
                "events": [],
                "_last_date": "",
                "_team_for_last": "",
                "_woba_numer": 0.0,
                "_woba_denom": 0,
            }

        p = players[bid]
        p["pa"] += 1
        if row.events in HIT_EVENTS:
            p["hits"] += 1
        if row.events in WALK_EVENTS:
            p["walks"] += 1

        p["_woba_numer"] += row.run_value
        if row.events not in WOBA_DENOM_EXCLUDE:
            p["_woba_denom"] += 1

        p["events"].append([
            row.game_date_str,
            row.events,
            round(row.run_value, 3),
            int(row.at_bat_number),
        ])

        if row.game_date_str >= p["_last_date"]:
            p["_last_date"] = row.game_date_str
            p["_team_for_last"] = row.batting_team

    lg_woba = weights["lgwOBA"]
    woba_scale = weights["wOBAScale"]
    lg_rpa = weights["lgRPA"]

    for p in players.values():
        p["team"] = p.pop("_team_for_last", "")
        p.pop("_last_date", None)

        denom = p.pop("_woba_denom")
        numer = p.pop("_woba_numer")
        woba = numer / denom if denom > 0 else 0.0
        p["woba"] = round(woba, 3)
        # wRC+ ≈ ( wRAA/PA + lgR/PA ) / lgR/PA * 100
        # Simplified form without park factor:
        # wRC+ = ((wOBA - lgwOBA) / wOBAScale + lgR/PA) / lgR/PA * 100
        wraa_per_pa = (woba - lg_woba) / woba_scale
        wrc_plus = ((wraa_per_pa + lg_rpa) / lg_rpa) * 100
        p["wrc_plus"] = round(wrc_plus, 0)

        p["events"].sort(key=lambda e: (e[0], e[3]))

    return sorted(players.values(), key=lambda p: p["hits"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch MLB PA data → JSON")
    parser.add_argument("--year", type=int, default=date.today().year,
                        help="Season year (default: current year)")
    args = parser.parse_args()
    year = args.year

    print(f"\n{'='*60}")
    print(f"  MLB Plate Appearance Fetcher — {year} season")
    print(f"{'='*60}\n")

    weights = get_weights(year)
    weights_year = year if year in WEIGHTS_BY_YEAR else max(WEIGHTS_BY_YEAR.keys())

    df = fetch_statcast(year)
    if df.empty:
        print("  No data returned. Is the season underway?")
        sys.exit(1)

    print("  Processing plate appearances …")
    player_list = build_player_json(df, weights)
    print(f"  Found {len(player_list):,} players with at least 1 PA.")

    readable_weights = {
        "out": 0.0,
        "walk": weights["wBB"],
        "hit_by_pitch": weights["wHBP"],
        "single": weights["w1B"],
        "double": weights["w2B"],
        "triple": weights["w3B"],
        "home_run": weights["wHR"],
    }

    output = {
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "season": year,
        "weights_source_year": weights_year,
        "linear_weights": readable_weights,
        "players": player_list,
    }

    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "player_data.json")
    with open(path, "w") as f:
        json.dump(output, f)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n  Wrote {path} ({size_mb:.1f} MB)")
    print(f"  To view: python -m http.server 8000  →  http://localhost:8000\n")


if __name__ == "__main__":
    main()
