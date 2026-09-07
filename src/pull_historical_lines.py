"""
Pull historical MLB totals lines from The Odds API.

2 snapshots per game day:
  - 10am ET (14:00 UTC) — decision-time line (when model runs)
  - 6pm ET (22:00 UTC) — closing line (just before first pitch)

Usage:
  python src/pull_historical_lines.py                # pull all years (2022-2026)
  python src/pull_historical_lines.py --year 2024    # pull one year
  python src/pull_historical_lines.py --start 2024 --end 2026

Requires ODDS_API_KEY environment variable.
Costs 10 credits per request (20 per game day).
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent.parent / "data" / "lines_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_DIR = RAW_DIR / "oddsapi_responses"
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

SEASON_STARTS = {
    2022: "2022-04-07",
    2023: "2023-03-30",
    2024: "2024-03-28",
    2025: "2025-03-27",
    2026: "2026-03-26",
}
SEASON_ENDS = {
    2022: "2022-10-05",
    2023: "2023-10-01",
    2024: "2024-09-30",
    2025: "2025-09-28",
    2026: "2026-09-30",
}

DECISION_TIME_UTC = "14:00:00"
CLOSING_TIME_UTC = "22:00:00"


def normalize_team(name: str) -> str:
    """Normalize team name for matching."""
    name = name.strip()
    aliases = {
        "athletics": "oakland athletics",
        "rays": "tampa bay rays",
        "dbacks": "arizona diamondbacks",
        "d-backs": "arizona diamondbacks",
        "red sox": "boston red sox",
        "white sox": "chicago white sox",
    }
    lower = name.lower()
    return aliases.get(lower, name)


def check_credits(api_key: str) -> dict:
    """Check remaining API credits."""
    url = "https://api.the-odds-api.com/v4/sports"
    resp = requests.get(url, params={"apiKey": api_key}, timeout=30)
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  Credits used: {used}, remaining: {remaining}")
    return {
        "remaining": int(remaining) if remaining != "?" else 0,
        "used": int(used) if used != "?" else 0,
    }


def save_metadata(years: list[int], credits_used: int) -> None:
    """Save fetch metadata for tracking."""
    from datetime import datetime
    meta_path = RAW_DIR / "oddsapi_metadata.json"
    existing = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing["last_fetch"] = datetime.now().isoformat()
    for year in years:
        existing[str(year)] = {
            "fetched_at": datetime.now().isoformat(),
            "source": "odds_api",
        }
    existing["total_credits_used"] = existing.get("total_credits_used", 0) + credits_used
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Metadata saved -> {meta_path}")


def fetch_snapshot(date_str: str, time_utc: str, label: str = "") -> dict:
    """Fetch historical odds snapshot for a given date and time.
    
    Stores raw JSON response for reprocessing (same pattern as NFL model).
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        raise ValueError("ODDS_API_KEY environment variable not set")

    timestamp = f"{date_str}T{time_utc}Z"
    url = (
        "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/"
        f"?apiKey={api_key}&regions=us&markets=totals&oddsFormat=american"
        f"&date={timestamp}"
    )

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            cost = resp.headers.get("x-requests-last", "?")
            print(f"  [remaining: {remaining}, used: {used}, cost: {cost}]")
            data = resp.json()

            # Store raw response (like NFL model's props_responses/)
            day_dir = RESPONSES_DIR / date_str[:7]  # e.g., "2024-07"
            day_dir.mkdir(exist_ok=True)
            raw_path = day_dir / f"{date_str}_{label}.json"
            with open(raw_path, "w") as f:
                json.dump(data, f)

            return data
        except requests.exceptions.HTTPError as e:
            body = ""
            if e.response is not None:
                try:
                    body = e.response.text[:200]
                except Exception:
                    pass
            if attempt + 1 < 3 and e.response is not None and e.response.status_code >= 500:
                delay = 2 * (2 ** attempt)
                print(f"  5xx error, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  HTTP {e.response.status_code}: {body}")
                raise
        except requests.exceptions.RequestException as e:
            if attempt + 1 < 3:
                delay = 2 * (2 ** attempt)
                print(f"  Request error, retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise

    return {"data": [], "timestamp": timestamp}


def extract_totals(snapshot: dict) -> dict:
    """Extract consensus totals line from all bookmakers in a snapshot.

    Returns: {(away_team, home_team): total_line}
    """
    results = {}
    for game in snapshot.get("data", []):
        away = game.get("away_team", "")
        home = game.get("home_team", "")

        totals = []
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Over" and "point" in outcome:
                            totals.append(outcome["point"])

        if totals:
            consensus = sorted(totals)[len(totals) // 2]
            results[(normalize_team(away), normalize_team(home))] = consensus

    return results


def pull_year(year: int, dry_run: bool = False) -> pd.DataFrame:
    """Pull all game days for a season."""
    start = SEASON_STARTS.get(year, f"{year}-03-25")
    end = SEASON_ENDS.get(year, f"{year}-10-01")

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    all_dates = []
    dt = start_dt
    while dt <= end_dt:
        all_dates.append(dt.strftime("%Y-%m-%d"))
        dt += timedelta(days=1)

    print(f"\n{'='*50}")
    print(f"{year}: {len(all_dates)} days ({start} to {end})")
    print(f"{'='*50}")

    rows = []
    requests_made = 0

    for i, date_str in enumerate(all_dates):
        cache_open = CACHE_DIR / f"hist_{date_str}_open.json"
        cache_close = CACHE_DIR / f"hist_{date_str}_close.json"

        # Decision-time snapshot (10am ET)
        if cache_open.exists():
            with open(cache_open) as f:
                open_snapshot = json.load(f)
            open_totals = extract_totals(open_snapshot)
        elif dry_run:
            print(f"  [{i+1}/{len(all_dates)}] {date_str} 10am ET — would pull")
            open_totals = {}
        else:
            print(f"  [{i+1}/{len(all_dates)}] {date_str} 10am ET...", end="")
            open_snapshot = fetch_snapshot(date_str, DECISION_TIME_UTC, "open")
            open_totals = extract_totals(open_snapshot)
            with open(cache_open, "w") as f:
                json.dump(open_snapshot, f)
            requests_made += 1
            time.sleep(1.5)

        # Closing snapshot (6pm ET)
        if cache_close.exists():
            with open(cache_close) as f:
                close_snapshot = json.load(f)
            close_totals = extract_totals(close_snapshot)
        elif dry_run:
            print(f"  [{i+1}/{len(all_dates)}] {date_str} 6pm ET — would pull")
            close_totals = {}
        else:
            print(f"  [{i+1}/{len(all_dates)}] {date_str} 6pm ET...", end="")
            close_snapshot = fetch_snapshot(date_str, CLOSING_TIME_UTC, "close")
            close_totals = extract_totals(close_snapshot)
            with open(cache_close, "w") as f:
                json.dump(close_snapshot, f)
            requests_made += 1
            time.sleep(1.5)

        all_games = set(open_totals.keys()) | set(close_totals.keys())
        for matchup in all_games:
            away, home = matchup
            rows.append({
                "date": date_str,
                "away_team": away,
                "home_team": home,
                "open_total": open_totals.get(matchup),
                "close_total": close_totals.get(matchup),
            })

        if all_games and (i + 1) % 10 == 0:
            print(f"    {len(all_games)} games found so far")

    df = pd.DataFrame(rows)
    if not dry_run:
        print(f"\n  {year}: {len(df)} rows, {requests_made} API requests ({requests_made * 10} credits)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Pull historical MLB totals from Odds API")
    parser.add_argument("--year", type=int, help="Pull a single year")
    parser.add_argument("--start", type=int, default=2022, help="Start year (default: 2022)")
    parser.add_argument("--end", type=int, default=2026, help="End year (default: 2026)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pulled without making API calls")
    args = parser.parse_args()

    years = [args.year] if args.year else list(range(args.start, args.end + 1))

    # Credit check before starting (like NFL model)
    if not args.dry_run:
        api_key = os.environ.get("ODDS_API_KEY", "")
        if not api_key:
            print("ERROR: ODDS_API_KEY environment variable not set.")
            return
        credits = check_credits(api_key)
        estimated_cost = len(years) * 400 * 20  # rough estimate: 400 game days per year, 20 credits per day
        if credits["remaining"] < estimated_cost * 0.5:
            print(f"  WARNING: Low credits ({credits['remaining']} remaining, ~{estimated_cost} estimated)")
            print(f"  Continue anyway? (Ctrl+C to abort)")
            time.sleep(5)

    all_dfs = []
    total_credits = 0

    for year in years:
        df = pull_year(year, dry_run=args.dry_run)
        if not df.empty:
            df["season"] = year
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = RAW_DIR / "lines_oddsapi.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n{'='*50}")
        print(f"Saved {len(combined)} rows to {out_path.name}")
        print(f"Seasons: {combined['season'].min()}-{combined['season'].max()}")
        print(f"With open line:  {combined['open_total'].notna().sum()}")
        print(f"With close line: {combined['close_total'].notna().sum()}")
        print(f"\nSample:")
        print(combined.head(10).to_string())

        # Save metadata (like NFL model)
        if not args.dry_run:
            save_metadata(years, total_credits)
    else:
        print("\nNo data pulled.")


if __name__ == "__main__":
    main()
