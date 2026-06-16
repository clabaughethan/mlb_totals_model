"""
Join all raw data sources into a single game-level feature matrix.
Outputs: data/processed/features.csv

One row per game with:
  - SP stats (FIP, xFIP, K/9, BB/9, IP/GS) for both pitchers
  - Rolling 14-day team offense (wRC+ proxy: R/G)
  - Rolling 14-day bullpen ERA
  - Park runs factor
  - Weather (temp, wind_mph, wind_out)
  - Rest days for each team
  - Target: total_runs
  - Join key: close_total (the line to beat)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Park run factors (100 = neutral, >100 = hitter-friendly)
# Source: FanGraphs 2018-2024 multi-year averages
PARK_FACTORS = {
    "Coors Field": 115,
    "Great American Ball Park": 105,
    "Citizens Bank Park": 104,
    "Fenway Park": 104,
    "Globe Life Field": 103,
    "Globe Life Park in Arlington": 106,
    "Guaranteed Rate Field": 103,
    "Yankee Stadium": 103,
    "Wrigley Field": 103,
    "Kauffman Stadium": 99,
    "Angel Stadium": 97,
    "Petco Park": 96,
    "Oracle Park": 95,
    "T-Mobile Park": 97,
    "Tropicana Field": 97,
    "Dodger Stadium": 98,
    "loanDepot park": 97,
    "Marlins Park": 97,
    "PNC Park": 98,
    "Target Field": 100,
    "Progressive Field": 100,
    "Truist Park": 101,
    "SunTrust Park": 101,
    "Nationals Park": 101,
    "Minute Maid Park": 100,
    "Busch Stadium": 98,
    "American Family Field": 100,
    "Chase Field": 101,
    "Oriole Park at Camden Yards": 101,
    "Rogers Centre": 100,
    "Comerica Park": 97,
    "Oakland Coliseum": 96,
    "Citi Field": 98,
    "Busch Stadium": 98,
    "Daikin Park": 100,
    "Sutter Health Park": 96,
    "UNIQLO Field at Dodger Stadium": 98,
}


def load_games() -> pd.DataFrame:
    combined = RAW_DIR / "games_all.csv"
    if combined.exists():
        df = pd.read_csv(combined, parse_dates=["date"], encoding="utf-8")
    else:
        # fetch_data.py still running — load completed season files
        frames = [pd.read_csv(p, parse_dates=["date"], encoding="utf-8") for p in sorted(RAW_DIR.glob("games_20??.csv"))]
        if not frames:
            raise FileNotFoundError("No game data found. Run fetch_data.py first.")
        df = pd.concat(frames, ignore_index=True)
        print(f"  (games_all.csv not ready yet — loaded {len(frames)} season files)")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_pitching() -> pd.DataFrame:
    frames = []
    for p in RAW_DIR.glob("pitching_stats_*.csv"):
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)

    # BRef columns: SO9, BB, HR, IP, ERA, WHIP, BAbip
    # Compute FIP = (13*HR + 3*BB - 2*SO) / IP + 3.10
    df["SO"] = df.get("SO", pd.Series(dtype=float))
    df["FIP"] = (13 * df["HR"] + 3 * df["BB"] - 2 * df["SO"]) / df["IP"].clip(lower=1) + 3.10
    df["FIP"] = df["FIP"].clip(lower=1.0, upper=8.0)

    # BB/9
    df["BB/9"] = df["BB"] / df["IP"].clip(lower=1) * 9

    # Rename BRef SO9 -> K/9 equivalent
    df["K/9"] = df.get("SO9", df["SO"] / df["IP"].clip(lower=1) * 9)

    df["ip_per_gs"] = df["IP"] / df["GS"].clip(lower=1)

    keep = ["Name", "season", "GS", "IP", "ERA", "FIP", "K/9", "BB/9", "ip_per_gs"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def load_team_batting() -> pd.DataFrame:
    frames = []
    for p in RAW_DIR.glob("team_batting_*.csv"):
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    return df


def load_weather() -> pd.DataFrame:
    path = RAW_DIR / "weather.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_lines() -> pd.DataFrame:
    path = RAW_DIR / "lines_all.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def build_rolling_offense(games: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute rolling R/G for each team over the last `window` calendar days.
    Returns a df indexed by (game_pk, team_id) with rolling_rpg.
    """
    # Build a per-team-per-game runs scored series
    home = games[["game_pk", "date", "home_team_id", "home_runs"]].rename(
        columns={"home_team_id": "team_id", "home_runs": "runs_scored"})
    away = games[["game_pk", "date", "away_team_id", "away_runs"]].rename(
        columns={"away_team_id": "team_id", "away_runs": "runs_scored"})
    team_games = pd.concat([home, away], ignore_index=True).sort_values("date")

    records = []
    for team_id, grp in team_games.groupby("team_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        # Rolling mean over last N games (not calendar days — simpler)
        grp["rolling_rpg"] = grp["runs_scored"].shift(1).rolling(window, min_periods=5).mean()
        records.append(grp[["game_pk", "team_id", "rolling_rpg"]])

    return pd.concat(records, ignore_index=True)


def build_rolling_bullpen(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Approximate rolling bullpen ERA per team using actual game runs allowed
    minus the SP's contribution (we don't have inning-level splits, so we
    use a rough proxy: runs allowed in games where SP went <6 IP).
    Better data can replace this later.
    Returns df with (game_pk, team_id, rolling_bullpen_era_proxy).
    """
    home = games[["game_pk", "date", "home_team_id", "away_runs"]].rename(
        columns={"home_team_id": "team_id", "away_runs": "runs_allowed"})
    away = games[["game_pk", "date", "away_team_id", "home_runs"]].rename(
        columns={"away_team_id": "team_id", "home_runs": "runs_allowed"})
    team_games = pd.concat([home, away], ignore_index=True).sort_values("date")

    records = []
    for team_id, grp in team_games.groupby("team_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["rolling_ra9"] = grp["runs_allowed"].shift(1).rolling(window, min_periods=5).mean() * 9 / 9
        records.append(grp[["game_pk", "team_id", "rolling_ra9"]])

    return pd.concat(records, ignore_index=True)


def build_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Compute days of rest for each team going into each game."""
    home = games[["game_pk", "date", "home_team_id"]].rename(columns={"home_team_id": "team_id"})
    away = games[["game_pk", "date", "away_team_id"]].rename(columns={"away_team_id": "team_id"})
    team_games = pd.concat([home, away], ignore_index=True).sort_values("date")

    records = []
    for team_id, grp in team_games.groupby("team_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["prev_game_date"] = grp["date"].shift(1)
        grp["rest_days"] = (
            pd.to_datetime(grp["date"]) - pd.to_datetime(grp["prev_game_date"])
        ).dt.days.fillna(3).clip(upper=7)
        records.append(grp[["game_pk", "team_id", "rest_days"]])

    return pd.concat(records, ignore_index=True)


def build_league_run_env(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute prior-season league average runs/game (both teams combined).
    Joined to each game so the model can adapt to year-over-year run environment shifts.
    Uses only the prior year — no leakage.
    """
    season_avg = (
        games.assign(season=pd.to_datetime(games["date"]).dt.year)
        .groupby("season")["total_runs"]
        .mean()
        .rename("lg_rpg")
        .reset_index()
    )
    # Shift forward: each game gets the PRIOR season's avg
    season_avg["season_join"] = season_avg["season"] + 1
    return season_avg[["season_join", "lg_rpg"]].rename(columns={"season_join": "season"})


def normalize_name(name: str) -> str:
    """Strip accents and lowercase for fuzzy name matching."""
    if not isinstance(name, str):
        return ""
    # Remove Unicode replacement characters from encoding corruption
    name = name.replace("�", "")
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def add_sp_stats(games: pd.DataFrame, pitching: pd.DataFrame) -> pd.DataFrame:
    """Join SP stats for home and away starters by normalized name + season."""
    games["season"] = pd.to_datetime(games["date"]).dt.year

    sp_cols = ["FIP", "K/9", "BB/9", "ip_per_gs", "ERA"]
    sp_cols = [c for c in sp_cols if c in pitching.columns]

    # Build lookup with normalized names
    # Deduplicate: traded pitchers appear once per team in BRef — keep row with most IP
    pitching = pitching.copy()
    pitching["name_norm"] = pitching["Name"].apply(normalize_name)
    pitching = (
        pitching.sort_values("IP", ascending=False)
        .drop_duplicates(subset=["name_norm", "season"], keep="first")
    )
    sp_lookup = pitching.set_index(["name_norm", "season"])[sp_cols]

    for side in ["home", "away"]:
        sp_name_col = f"{side}_sp_name"
        games[f"{side}_sp_name_norm"] = games[sp_name_col].apply(normalize_name)
        for stat in sp_cols:
            col = f"{side}_sp_{stat.replace('/', '_').lower()}"
            games[col] = games.apply(
                lambda r, s=stat: sp_lookup.loc[(r[f"{side}_sp_name_norm"], r["season"]), s]
                if (r[f"{side}_sp_name_norm"], r["season"]) in sp_lookup.index else np.nan,
                axis=1,
            )
        games = games.drop(columns=[f"{side}_sp_name_norm"])

    return games


def main():
    print("Loading raw data...")
    games = load_games()
    pitching = load_pitching()
    weather = load_weather()
    lines = load_lines()

    print(f"  {len(games)} games, {len(pitching)} pitcher-seasons, "
          f"{len(weather)} weather rows, {len(lines)} line rows")

    print("Building rolling offense...")
    rolling_off = build_rolling_offense(games)

    print("Building rolling bullpen proxy...")
    rolling_bp = build_rolling_bullpen(games)

    print("Building rest days...")
    rest = build_rest_days(games)

    print("Joining SP stats...")
    games = add_sp_stats(games, pitching)

    # Join rolling offense for home and away
    rolling_off_home = rolling_off.rename(
        columns={"team_id": "home_team_id", "rolling_rpg": "home_rolling_rpg"})
    rolling_off_away = rolling_off.rename(
        columns={"team_id": "away_team_id", "rolling_rpg": "away_rolling_rpg"})
    games = games.merge(rolling_off_home[["game_pk", "home_team_id", "home_rolling_rpg"]],
                        on=["game_pk", "home_team_id"], how="left")
    games = games.merge(rolling_off_away[["game_pk", "away_team_id", "away_rolling_rpg"]],
                        on=["game_pk", "away_team_id"], how="left")

    # Join rolling bullpen
    bp_home = rolling_bp.rename(columns={"team_id": "home_team_id", "rolling_ra9": "home_rolling_ra9"})
    bp_away = rolling_bp.rename(columns={"team_id": "away_team_id", "rolling_ra9": "away_rolling_ra9"})
    games = games.merge(bp_home[["game_pk", "home_team_id", "home_rolling_ra9"]],
                        on=["game_pk", "home_team_id"], how="left")
    games = games.merge(bp_away[["game_pk", "away_team_id", "away_rolling_ra9"]],
                        on=["game_pk", "away_team_id"], how="left")

    # Join rest days
    rest_home = rest.rename(columns={"team_id": "home_team_id", "rest_days": "home_rest_days"})
    rest_away = rest.rename(columns={"team_id": "away_team_id", "rest_days": "away_rest_days"})
    games = games.merge(rest_home[["game_pk", "home_team_id", "home_rest_days"]],
                        on=["game_pk", "home_team_id"], how="left")
    games = games.merge(rest_away[["game_pk", "away_team_id", "away_rest_days"]],
                        on=["game_pk", "away_team_id"], how="left")

    # Prior-season league run environment
    print("Building league run environment...")
    league_env = build_league_run_env(games)
    games = games.merge(league_env, on="season", how="left")

    # Park factor
    games["park_factor"] = games["venue_name"].map(PARK_FACTORS).fillna(100)

    # Weather
    if not weather.empty:
        wx_cols = ["game_pk", "temp_f", "wind_mph", "wind_out", "is_dome"]
        games = games.merge(weather[wx_cols], on="game_pk", how="left")
        games["wind_out"] = games["wind_out"].fillna(False)
        games["is_dome"] = games["is_dome"].fillna(False)

    # Betting lines
    if not lines.empty:
        lines_join = lines[["date", "home_team", "away_team", "open_total", "close_total"]].copy()
        lines_join["date"] = pd.to_datetime(lines_join["date"])
        games["date"] = pd.to_datetime(games["date"])
        games = games.merge(lines_join, on=["date", "home_team", "away_team"], how="left")

    # Save
    out_path = PROCESSED_DIR / "features.csv"
    games.to_csv(out_path, index=False)
    print(f"\nFeature matrix saved: {len(games)} rows x {len(games.columns)} cols -> {out_path}")

    # Coverage report
    feature_cols = [c for c in games.columns if c not in
                    ["game_pk", "date", "home_team", "away_team", "home_sp_name", "away_sp_name",
                     "venue_name", "game_time_utc", "season"]]
    print("\nFeature coverage (% non-null):")
    coverage = games[feature_cols].notna().mean().sort_values()
    print(coverage.to_string())


if __name__ == "__main__":
    main()
