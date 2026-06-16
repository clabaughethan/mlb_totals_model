"""
Fetch historical MLB game results with starting pitchers via statsapi.
Outputs: data/raw/games_{year}.csv for each season.

Each row = one game with:
  game_pk, date, home_team, away_team,
  home_runs, away_runs, total_runs,
  home_sp_id, home_sp_name, away_sp_id, away_sp_name,
  venue_id, venue_name, game_time_utc
"""

import statsapi
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = list(range(2015, 2027))  # 2015-2026


def get_starting_pitcher(game_pk: int, side: str) -> tuple[int | None, str | None]:
    """Return (player_id, name) for the starting pitcher on 'home' or 'away' side."""
    try:
        boxscore = statsapi.boxscore_data(game_pk)
        pitchers = boxscore[side]["pitchers"]
        if not pitchers:
            return None, None
        # First pitcher listed is the starter
        sp_id = pitchers[0]
        sp_info = boxscore["playerInfo"].get(f"ID{sp_id}", {})
        return sp_id, sp_info.get("fullName")
    except Exception:
        return None, None


def fetch_season(year: int) -> pd.DataFrame:
    out_path = RAW_DIR / f"games_{year}.csv"
    if out_path.exists():
        print(f"{year}: already fetched, skipping.")
        return pd.read_csv(out_path)

    print(f"\nFetching {year} schedule...")
    schedule = statsapi.schedule(
        start_date=f"{year}-03-20",
        end_date=f"{year}-11-05",
        sportId=1,  # MLB
    )

    # Only completed regular season games
    games = [g for g in schedule if g["status"] == "Final" and g["game_type"] == "R"]
    print(f"  {len(games)} completed regular season games found.")

    rows = []
    for g in tqdm(games, desc=f"  {year} games"):
        game_pk = g["game_id"]
        home_sp_id, home_sp_name = get_starting_pitcher(game_pk, "home")
        away_sp_id, away_sp_name = get_starting_pitcher(game_pk, "away")

        rows.append({
            "game_pk": game_pk,
            "date": g["game_date"],
            "game_time_utc": g.get("game_datetime"),
            "venue_id": g.get("venue_id"),
            "venue_name": g.get("venue_name"),
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "home_team_id": g["home_id"],
            "away_team_id": g["away_id"],
            "home_runs": g["home_score"],
            "away_runs": g["away_score"],
            "total_runs": g["home_score"] + g["away_score"],
            "home_sp_id": home_sp_id,
            "home_sp_name": home_sp_name,
            "away_sp_id": away_sp_id,
            "away_sp_name": away_sp_name,
        })

        # Be polite to the API
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path.name}")
    return df


def main():
    all_seasons = []
    for year in SEASONS:
        df = fetch_season(year)
        all_seasons.append(df)

    combined = pd.concat(all_seasons, ignore_index=True)
    combined_path = RAW_DIR / "games_all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined dataset: {len(combined)} games -> {combined_path}")

    print("\nSample:")
    print(combined[["date", "home_team", "away_team", "total_runs",
                     "home_sp_name", "away_sp_name"]].head(10).to_string())


if __name__ == "__main__":
    main()
