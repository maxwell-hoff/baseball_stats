#!/usr/bin/env python3
"""
Fetch park factors from Baseball Savant (Statcast) and export to JSON.

Scrapes the Statcast Park Factors leaderboard for per-event-type
(1B, 2B, 3B, HR, BB, SO) and per-handedness (L, R) factors.

Usage:
    python park_factors.py                  # current year, 3-year rolling
    python park_factors.py --year 2025      # specific year
    python park_factors.py --rolling 1      # 1-year window
"""

import argparse
import json
import os
import re
import sys
from datetime import date

import requests

# ══════════════════════════════════════════════════════════════════════
# CONFIGURABLE DEFAULTS
# ══════════════════════════════════════════════════════════════════════
DEFAULT_YEAR = date.today().year
DEFAULT_ROLLING = 3

BASE_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?type=year&year={year}&batSide={side}&stat=index_wOBA"
    "&condition=All&rolling={rolling}&parks=mlb"
)

STAT_COLUMNS = {
    "index_wOBA":  "wOBA",
    "index_woba":  "wOBA",
    "index_1B":    "1B",
    "index_1b":    "1B",
    "index_2B":    "2B",
    "index_2b":    "2B",
    "index_3B":    "3B",
    "index_3b":    "3B",
    "index_HR":    "HR",
    "index_hr":    "HR",
    "index_BB":    "BB",
    "index_bb":    "BB",
    "index_SO":    "SO",
    "index_so":    "SO",
}

# Map Savant team abbreviations to the abbreviations used in our Statcast
# event data (from pybaseball). Most are identical; only known mismatches
# need entries here.
SAVANT_TO_STATCAST = {
    "ARI": "AZ",
    "OAK": "ATH",
}

# Reverse map for printing/comparison
VENUE_TO_TEAM = {
    "Angel Stadium": "LAA",
    "Busch Stadium": "STL",
    "Chase Field": "AZ",
    "Citi Field": "NYM",
    "Citizens Bank Park": "PHI",
    "Comerica Park": "DET",
    "Coors Field": "COL",
    "Dodger Stadium": "LAD",
    "UNIQLO Field at Dodger Stadium": "LAD",
    "Fenway Park": "BOS",
    "Globe Life Field": "TEX",
    "Great American Ball Park": "CIN",
    "Guaranteed Rate Field": "CWS",
    "Rate Field": "CWS",
    "Kauffman Stadium": "KC",
    "loanDepot park": "MIA",
    "LoanDepot Park": "MIA",
    "Marlins Park": "MIA",
    "Minute Maid Park": "HOU",
    "Daikin Park": "HOU",
    "Nationals Park": "WSH",
    "Oracle Park": "SF",
    "Oriole Park at Camden Yards": "BAL",
    "Petco Park": "SD",
    "PNC Park": "PIT",
    "Progressive Field": "CLE",
    "Rogers Centre": "TOR",
    "T-Mobile Park": "SEA",
    "Target Field": "MIN",
    "Tropicana Field": "TB",
    "Truist Park": "ATL",
    "Wrigley Field": "CHC",
    "Yankee Stadium": "NYY",
    "American Family Field": "MIL",
    "Sutter Health Park": "ATH",
}


def fetch_savant_park_factors(year: int, bat_side: str, rolling: int) -> list[dict]:
    """Fetch park factors from Baseball Savant for a given batSide.

    bat_side: '' (both), 'L', or 'R'
    Returns the parsed JSON data array.
    """
    url = BASE_URL.format(year=year, side=bat_side, rolling=rolling)
    print(f"  Fetching: batSide={bat_side or 'Both'} year={year} rolling={rolling}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    match = re.search(r"var defined_data\s*=\s*(.*?);", resp.text, re.DOTALL)
    if not match:
        match = re.search(r"data\s*=\s*(\[.*?\]);", resp.text, re.DOTALL)
    if not match:
        print(f"    ERROR: Could not find embedded data in page HTML.")
        print(f"    Page length: {len(resp.text)} chars")
        return []

    raw = match.group(1).strip()
    data = json.loads(raw)
    print(f"    Got {len(data)} venues")
    return data


CLUB_NAME_TO_ABBREV = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH",
    "Blue Jays": "TOR", "Braves": "ATL", "Brewers": "MIL",
    "Cardinals": "STL", "Cubs": "CHC", "D-backs": "AZ",
    "Diamondbacks": "AZ", "Dodgers": "LAD", "Giants": "SF",
    "Guardians": "CLE", "Indians": "CLE",
    "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM",
    "Nationals": "WSH", "Orioles": "BAL", "Padres": "SD",
    "Phillies": "PHI", "Pirates": "PIT", "Rangers": "TEX",
    "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
    "Rockies": "COL", "Royals": "KC", "Tigers": "DET",
    "Twins": "MIN", "White Sox": "CWS", "Yankees": "NYY",
}


def resolve_team_abbrev(row: dict) -> str | None:
    """Extract the team abbreviation from a Savant data row."""
    for key in ("team_abbrev", "team_name_abbrev", "team_abbr", "team"):
        val = row.get(key)
        if val and isinstance(val, str) and len(val) <= 4:
            return SAVANT_TO_STATCAST.get(val, val)

    venue = row.get("venue_name", "")
    if venue in VENUE_TO_TEAM:
        return VENUE_TO_TEAM[venue]

    club = row.get("name_display_club", "")
    if club in CLUB_NAME_TO_ABBREV:
        return CLUB_NAME_TO_ABBREV[club]

    return None


def extract_factors(row: dict) -> dict:
    """Extract the stat factors we care about from a data row."""
    factors = {}
    for col, label in STAT_COLUMNS.items():
        val = row.get(col)
        if val is not None:
            try:
                factors[label] = int(round(float(val)))
            except (ValueError, TypeError):
                pass
    return factors


def build_park_factors(year: int, rolling: int) -> dict:
    """Build the full park factors dict: {hand: {team: {stat: factor}}}."""
    sides = [("", "All"), ("L", "L"), ("R", "R")]
    result = {}

    for side_param, side_label in sides:
        data = fetch_savant_park_factors(year, side_param, rolling)
        if not data:
            continue

        if side_label == "All" and data:
            print(f"\n    Available columns: {sorted(data[0].keys())}\n")

        teams = {}
        for row in data:
            abbrev = resolve_team_abbrev(row)
            if not abbrev:
                venue = row.get("venue_name", "???")
                print(f"    WARNING: Could not resolve team for venue '{venue}'")
                continue

            factors = extract_factors(row)
            if factors:
                teams[abbrev] = factors

        result[side_label] = teams
        print(f"    Mapped {len(teams)} teams for {side_label}")

    return result


def export_json(park_factors: dict, path: str = "data/park_factors.json"):
    """Export park factors to JSON for use by the website."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(park_factors, f, indent=2)

    print(f"\n{'═' * 70}")
    print(f"  EXPORTED")
    print(f"{'═' * 70}\n")
    print(f"  Wrote {path}")
    print(f"  Keys: {list(park_factors.keys())}")
    for hand, teams in park_factors.items():
        print(f"  {hand}: {len(teams)} teams")
    print()


def print_factors(park_factors: dict):
    """Print a nicely formatted summary of the park factors."""
    for hand in ["All", "L", "R"]:
        teams = park_factors.get(hand, {})
        if not teams:
            continue

        label = {"All": "OVERALL", "L": "LEFT-HANDED BATTERS", "R": "RIGHT-HANDED BATTERS"}[hand]
        print(f"\n{'═' * 70}")
        print(f"  {label}")
        print(f"{'═' * 70}\n")

        header = f"  {'Team':<6} {'wOBA':>6} {'1B':>6} {'2B':>6} {'3B':>6} {'HR':>6} {'BB':>6} {'SO':>6}"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for team in sorted(teams.keys()):
            f = teams[team]
            print(f"  {team:<6} {f.get('wOBA', '-'):>6} {f.get('1B', '-'):>6} "
                  f"{f.get('2B', '-'):>6} {f.get('3B', '-'):>6} {f.get('HR', '-'):>6} "
                  f"{f.get('BB', '-'):>6} {f.get('SO', '-'):>6}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Fetch Statcast park factors from Baseball Savant")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR,
                        help=f"Year to fetch (default: {DEFAULT_YEAR})")
    parser.add_argument("--rolling", type=int, default=DEFAULT_ROLLING, choices=[1, 2, 3],
                        help=f"Rolling window in years (default: {DEFAULT_ROLLING})")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  Baseball Savant Park Factors")
    print(f"  Year: {args.year} | Rolling: {args.rolling}-year window")
    print(f"{'=' * 70}\n")

    print("─── Fetching from Baseball Savant ─────────────────────────\n")
    park_factors = build_park_factors(args.year, args.rolling)

    if not park_factors:
        print("  ERROR: No data retrieved. Check network and URL.")
        sys.exit(1)

    print_factors(park_factors)
    export_json(park_factors)


if __name__ == "__main__":
    main()
