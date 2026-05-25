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
from datetime import date, datetime

import pandas as pd
import pybaseball


# ── wOBA linear weights + league constants (FanGraphs Guts!) ─────────
# Updated yearly — https://www.fangraphs.com/tools/guts
WEIGHTS_BY_YEAR = {
    2023: {
        "wBB": 0.696, "wHBP": 0.726, "w1B": 0.883,
        "w2B": 1.244, "w3B": 1.569, "wHR": 2.015,
        "lgwOBA": 0.318, "wOBAScale": 1.204, "lgRPA": 0.119,
    },
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

EVENT_WEIGHT_KEY = {
    "single":       "w1B",
    "double":       "w2B",
    "triple":       "w3B",
    "home_run":     "wHR",
    "walk":         "wBB",
    "intent_walk":  "wBB",
    "hit_by_pitch": "wHBP",
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


def build_player_json(all_pa: pd.DataFrame) -> list[dict]:
    batter_ids = all_pa["batter"].unique().tolist()
    print(f"  Looking up names for {len(batter_ids):,} unique batters …")
    name_df = pybaseball.playerid_reverse_lookup(batter_ids, key_type="mlbam")
    batter_names: dict[int, str] = {}
    for _, row in name_df.iterrows():
        first = str(row["name_first"]).strip().title()
        last = str(row["name_last"]).strip().title()
        batter_names[int(row["key_mlbam"])] = f"{first} {last}"

    players: dict[int, dict] = {}
    for row in all_pa.itertuples(index=False):
        bid = int(row.batter)
        if bid not in players:
            players[bid] = {
                "name": batter_names.get(bid, f"Unknown ({bid})"),
                "team": "",
                "events": [],
                "_last_date": "",
                "_team_for_last": "",
            }

        p = players[bid]
        team = row.away_team if row.inning_topbot == "Top" else row.home_team
        xw = row.estimated_woba_using_speedangle
        stand = row.stand if hasattr(row, "stand") and pd.notna(row.stand) else None
        p["events"].append([
            row.game_date_str,   # [0]
            row.events,          # [1]
            round(row.run_value, 3),  # [2]
            int(row.at_bat_number),   # [3]
            team,                # [4] batting team
            round(xw, 3) if pd.notna(xw) else None,  # [5] xwOBA
            row.home_team,       # [6] stadium (home team of game)
            stand,               # [7] batter handedness (L/R)
        ])

        if row.game_date_str >= p["_last_date"]:
            p["_last_date"] = row.game_date_str
            p["_team_for_last"] = team

    for p in players.values():
        p["team"] = p.pop("_team_for_last", "")
        p.pop("_last_date", None)
        p["events"].sort(key=lambda e: (e[0], e[3]))

    return list(players.values())


def main():
    parser = argparse.ArgumentParser(description="Fetch MLB PA data → JSON")
    parser.add_argument(
        "--years",
        help="Comma-separated years to fetch (default: current + 2 prior)",
    )
    args = parser.parse_args()

    if args.years:
        years = sorted(int(y) for y in args.years.split(","))
    else:
        current = date.today().year
        years = [current - 2, current - 1, current]

    print(f"\n{'='*60}")
    print(f"  MLB Plate Appearance Fetcher — {', '.join(map(str, years))}")
    print(f"{'='*60}\n")

    pa_frames: list[pd.DataFrame] = []
    for year in years:
        weights = get_weights(year)
        df = fetch_season(year)
        if df.empty:
            print(f"    No data for {year}, skipping.")
            continue
        pa = df[df["events"].isin(PA_EVENTS)].copy()
        pa["run_value"] = pa["events"].apply(lambda e: run_value(e, weights))
        pa["game_date_str"] = pd.to_datetime(pa["game_date"]).dt.strftime("%Y-%m-%d")
        pa_frames.append(pa)
        print(f"    {len(pa):,} plate appearances")

    if not pa_frames:
        print("  No data returned for any year. Exiting.")
        sys.exit(1)

    all_pa = pd.concat(pa_frames, ignore_index=True)
    print(f"\n  Total: {len(all_pa):,} plate appearances across {len(pa_frames)} season(s)")

    player_list = build_player_json(all_pa)
    print(f"  {len(player_list):,} unique players")

    league_constants: dict[str, dict] = {}
    for year in years:
        w = get_weights(year)
        league_constants[str(year)] = {
            "lgwOBA": w["lgwOBA"],
            "wOBAScale": w["wOBAScale"],
            "lgRPA": w["lgRPA"],
        }

    output = {
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "years_available": years,
        "league_constants": league_constants,
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
