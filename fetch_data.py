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

    runner_counts: dict[int, dict] = defaultdict(lambda: {"sb": 0, "cs": 0})
    runner_event_rows: dict[int, list] = defaultdict(list)

    for mask_series, evt_type in [(sb_mask, "sb"), (cs_mask, "cs")]:
        subset = full_df[mask_series]
        if subset.empty:
            continue
        sub_desc = subset[desc_col].str.lower()
        to_2nd = sub_desc.str.contains("2nd|second", na=False, regex=True)
        to_3rd = sub_desc.str.contains("3rd|third", na=False, regex=True)
        to_home = sub_desc.str.contains("home|scores", na=False, regex=True)

        rv = SB_RUN_VALUE if evt_type == "sb" else CS_RUN_VALUE
        evt_name = "stolen_base" if evt_type == "sb" else "caught_stealing"

        for cond, col in [(to_2nd, "on_1b"), (to_3rd, "on_2b"), (to_home, "on_3b")]:
            matched = subset[cond]
            for _, row in matched.iterrows():
                runner_id = row.get(col)
                if pd.isna(runner_id):
                    continue
                rid = int(runner_id)
                runner_counts[rid][evt_type] += 1
                game_date = pd.to_datetime(row["game_date"]).strftime("%Y-%m-%d")
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                topbot = row.get("inning_topbot", "")
                team = away if topbot == "Top" else home
                ab_num = int(row["at_bat_number"]) if pd.notna(row.get("at_bat_number")) else 0
                runner_event_rows[rid].append([
                    game_date, evt_name, round(rv, 3), ab_num,
                    team, None, home, None,
                ])

    total_sb = sum(v["sb"] for v in runner_counts.values())
    total_cs = sum(v["cs"] for v in runner_counts.values())
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

        sb = runner_counts[bid]["sb"]
        cs = runner_counts[bid]["cs"]
        wsb = (sb * SB_RUN_VALUE + cs * CS_RUN_VALUE) - (lg_wsb_per_pa * pa_count)

        opps = (group["on_1b"].notna() & (group["outs_when_up"] < 2)).sum()
        gdps = group["events"].isin(GDP_EVENTS).sum()
        if opps > 0:
            player_gdp_rate = gdps / opps
            wgdp = (lg_gdp_rate - player_gdp_rate) * opps * GDP_EXTRA_COST
        else:
            wgdp = 0.0

        evts = sorted(runner_event_rows.get(bid, []), key=lambda e: (e[0], e[3]))
        bsr_dict[bid] = {
            "bsr": round(float(wsb + wgdp), 2),
            "sb": int(sb),
            "cs": int(cs),
            "baserunning_events": evts,
        }

    return bsr_dict


def build_player_json(all_pa: pd.DataFrame) -> tuple[list[dict], dict[str, int]]:
    """Returns (player_list, name_to_id_map)."""
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
        bs = getattr(row, "bat_score", None)
        pbs = getattr(row, "post_bat_score", None)
        actual_r = int(pbs - bs) if pd.notna(bs) and pd.notna(pbs) else 0
        p["events"].append([
            row.game_date_str,   # [0]
            row.events,          # [1]
            round(row.run_value, 3),  # [2]
            int(row.at_bat_number),   # [3]
            team,                # [4] batting team
            round(xw, 3) if pd.notna(xw) else None,  # [5] xwOBA
            row.home_team,       # [6] stadium (home team of game)
            stand,               # [7] batter handedness (L/R)
            actual_r,            # [8] actual runs scored on this PA
        ])

        if row.game_date_str >= p["_last_date"]:
            p["_last_date"] = row.game_date_str
            p["_team_for_last"] = team

    name_to_id: dict[str, int] = {}
    for bid, p in players.items():
        p["team"] = p.pop("_team_for_last", "")
        p.pop("_last_date", None)
        p["events"].sort(key=lambda e: (e[0], e[3]))
        name_to_id[p["name"]] = bid

    return list(players.values()), name_to_id


def build_pitcher_json(all_pa: pd.DataFrame) -> list[dict]:
    """Build pitcher records — same event format as batters, grouped by pitcher."""
    pitcher_ids = all_pa["pitcher"].dropna().unique().astype(int).tolist()
    print(f"  Looking up names for {len(pitcher_ids):,} unique pitchers …")
    name_df = pybaseball.playerid_reverse_lookup(pitcher_ids, key_type="mlbam")
    pitcher_names: dict[int, str] = {}
    for _, row in name_df.iterrows():
        first = str(row["name_first"]).strip().title()
        last = str(row["name_last"]).strip().title()
        pitcher_names[int(row["key_mlbam"])] = f"{first} {last}"

    pitchers: dict[int, dict] = {}
    for row in all_pa.itertuples(index=False):
        pid = int(row.pitcher) if pd.notna(row.pitcher) else None
        if pid is None:
            continue
        if pid not in pitchers:
            pitchers[pid] = {
                "name": pitcher_names.get(pid, f"Unknown ({pid})"),
                "team": "",
                "playerType": "pitcher",
                "events": [],
                "_last_date": "",
                "_team_for_last": "",
            }

        p = pitchers[pid]
        team = row.home_team if row.inning_topbot == "Top" else row.away_team
        xw = row.estimated_woba_using_speedangle
        stand = row.stand if hasattr(row, "stand") and pd.notna(row.stand) else None
        bs = getattr(row, "bat_score", None)
        pbs = getattr(row, "post_bat_score", None)
        actual_r = int(pbs - bs) if pd.notna(bs) and pd.notna(pbs) else 0
        p["events"].append([
            row.game_date_str,
            row.events,
            round(row.run_value, 3),
            int(row.at_bat_number),
            team,
            round(xw, 3) if pd.notna(xw) else None,
            row.home_team,
            stand,
            actual_r,            # [8] actual runs scored on this PA (allowed by pitcher)
        ])

        if row.game_date_str >= p["_last_date"]:
            p["_last_date"] = row.game_date_str
            p["_team_for_last"] = team

    for pid, p in pitchers.items():
        p["team"] = p.pop("_team_for_last", "")
        p.pop("_last_date", None)
        p["events"].sort(key=lambda e: (e[0], e[3]))
        p["bsr"] = 0.0
        p["sb"] = 0
        p["cs"] = 0
        p["baserunning_events"] = []
        p["def_runs"] = 0.0

    return list(pitchers.values())


def fetch_fielding_runs(years: list[int]) -> dict[int, float]:
    """Fetch Defensive Runs Above Average (Def) from FanGraphs batting leaderboards.

    Returns {mlbam_id: cumulative_def_runs} across all requested years.
    """
    fg_def: dict[int, float] = {}

    for year in years:
        print(f"    Fetching FanGraphs batting leaderboard for {year} …")
        try:
            fg = pybaseball.batting_stats(year, qual=0)
            if fg.empty:
                print(f"      Empty result for {year}, skipping.")
                continue
            if "Def" not in fg.columns or "IDfg" not in fg.columns:
                print(f"      Missing Def/IDfg columns for {year}, skipping.")
                continue
            for _, row in fg.iterrows():
                fg_id = row.get("IDfg")
                def_val = row.get("Def", 0)
                if pd.notna(fg_id) and pd.notna(def_val):
                    fg_id = int(fg_id)
                    fg_def[fg_id] = fg_def.get(fg_id, 0.0) + float(def_val)
        except Exception as e:
            print(f"      Warning: could not fetch for {year}: {e}")

    if not fg_def:
        print("    No fielding data found.")
        return {}

    fg_ids = list(fg_def.keys())
    print(f"    Mapping {len(fg_ids):,} FanGraphs IDs → MLBAM …")
    try:
        mapping = pybaseball.playerid_reverse_lookup(fg_ids, key_type="fangraphs")
    except Exception as e:
        print(f"    Warning: ID mapping failed: {e}")
        return {}

    fg_to_mlbam: dict[int, int] = {}
    for _, row in mapping.iterrows():
        fid = row.get("key_fangraphs")
        mid = row.get("key_mlbam")
        if pd.notna(fid) and pd.notna(mid):
            fg_to_mlbam[int(fid)] = int(mid)

    result: dict[int, float] = {}
    for fg_id, def_val in fg_def.items():
        mlbam_id = fg_to_mlbam.get(fg_id)
        if mlbam_id:
            result[mlbam_id] = round(def_val, 2)

    print(f"    Mapped fielding data for {len(result):,} players.")
    return result


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

    raw_frames: list[pd.DataFrame] = []
    pa_frames: list[pd.DataFrame] = []
    for year in years:
        weights = get_weights(year)
        df = fetch_season(year)
        if df.empty:
            print(f"    No data for {year}, skipping.")
            continue
        raw_frames.append(df)
        pa = df[df["events"].isin(PA_EVENTS)].copy()
        pa["run_value"] = pa["events"].apply(lambda e: run_value(e, weights))
        pa["game_date_str"] = pd.to_datetime(pa["game_date"]).dt.strftime("%Y-%m-%d")
        pa_frames.append(pa)
        print(f"    {len(pa):,} plate appearances")

    if not pa_frames:
        print("  No data returned for any year. Exiting.")
        sys.exit(1)

    all_raw = pd.concat(raw_frames, ignore_index=True)
    all_pa = pd.concat(pa_frames, ignore_index=True)
    print(f"\n  Total: {len(all_pa):,} plate appearances across {len(pa_frames)} season(s)")

    print(f"\n  Computing BsR (wSB + wGDP) …")
    bsr_data = compute_bsr(all_raw, all_pa)
    del all_raw

    print(f"\n  Fetching fielding data (Defensive Runs) …")
    def_data = fetch_fielding_runs(years)

    player_list, id_map = build_player_json(all_pa)
    print(f"  {len(player_list):,} unique batters")

    for p in player_list:
        bid = id_map.get(p["name"])
        b = bsr_data.get(bid, {})
        p["bsr"] = b.get("bsr", 0.0)
        p["sb"] = b.get("sb", 0)
        p["cs"] = b.get("cs", 0)
        p["baserunning_events"] = b.get("baserunning_events", [])
        p["def_runs"] = def_data.get(bid, 0.0)
        p["playerType"] = "batter"

    print(f"\n  Building pitcher records …")
    pitcher_list = build_pitcher_json(all_pa)
    print(f"  {len(pitcher_list):,} unique pitchers")

    all_players = player_list + pitcher_list

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
        "players": all_players,
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
