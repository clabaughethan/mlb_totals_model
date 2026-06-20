"""
Live prediction script for today's MLB game totals.

Usage:
  python src/predict_today.py              # today's games
  python src/predict_today.py --date 2025-06-20

Outputs a table of today's games with model predictions, current line (if available),
and whether the model sees an edge worth betting.

Requirements:
  - models/ridge_production.joblib  (run train.py first)
  - Odds API key in env var ODDS_API_KEY (optional — skipped if absent)
"""

import argparse
import os
import random
import sys
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
import statsapi
import joblib


def _retry(fn, *args, max_attempts=3, base_delay=2, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on 5xx errors."""
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            if attempt + 1 < max_attempts and e.response is not None and e.response.status_code >= 500:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  statsapi 5xx (attempt {attempt+1}/{max_attempts}), retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                raise


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
MODELS_DIR = Path(__file__).parent.parent / "models"

MIN_EDGE = 0.75
MAX_EDGE = 1.5

PARK_INFO = {
    "Oriole Park at Camden Yards":   (39.2839, -76.6218, 230),
    "Fenway Park":                    (42.3467, -71.0972, 270),
    "Yankee Stadium":                 (40.8296, -73.9262, 0),
    "Rogers Centre":                  (43.6414, -79.3894, 0),
    "Guaranteed Rate Field":          (41.8300, -87.6338, 135),
    "Wrigley Field":                  (41.9484, -87.6553, 315),
    "Progressive Field":              (41.4962, -81.6852, 225),
    "Comerica Park":                  (42.3390, -83.0485, 335),
    "Kauffman Stadium":               (39.0517, -94.4803, 0),
    "Target Field":                   (44.9817, -93.2781, 0),
    "Tropicana Field":                (27.7682, -82.6534, 0),
    "Globe Life Field":               (32.7473, -97.0824, 0),
    "Minute Maid Park":               (29.7573, -95.3555, 0),
    "Oakland Coliseum":               (37.7516, -122.2005, 270),
    "T-Mobile Park":                  (47.5914, -122.3325, 315),
    "Angel Stadium":                  (33.8003, -117.8827, 225),
    "Dodger Stadium":                 (34.0739, -118.2400, 315),
    "Petco Park":                     (32.7076, -117.1570, 270),
    "Oracle Park":                    (37.7786, -122.3893, 270),
    "Coors Field":                    (39.7559, -104.9942, 315),
    "Chase Field":                    (33.4455, -112.0667, 0),
    "Busch Stadium":                  (38.6226, -90.1928, 315),
    "American Family Field":          (43.0280, -87.9712, 0),
    "PNC Park":                       (40.4469, -80.0057, 135),
    "Great American Ball Park":       (39.0979, -84.5082, 225),
    "Truist Park":                    (33.8908, -84.4678, 0),
    "Nationals Park":                 (38.8730, -77.0074, 270),
    "loanDepot park":                 (25.7781, -80.2197, 0),
    "Citi Field":                     (40.7571, -73.8458, 0),
    "Citizens Bank Park":             (39.9061, -75.1665, 180),
    "Globe Life Park in Arlington":   (32.7512, -97.0832, 0),
    "SunTrust Park":                  (33.8908, -84.4678, 0),
    "Marlins Park":                   (25.7781, -80.2197, 0),
    # Renamed venues
    "Daikin Park":                    (29.7573, -95.3555, 0),   # fka Minute Maid Park, retractable
    "Sutter Health Park":             (38.5727, -121.4944, 315), # Athletics temp home, Sacramento
    "UNIQLO Field at Dodger Stadium": (34.0739, -118.2400, 315),
}

DOMES = {
    "Tropicana Field", "Chase Field", "Minute Maid Park", "Daikin Park",
    "Globe Life Field", "loanDepot park", "Marlins Park", "Rogers Centre",
}

PARK_FACTORS = {
    "Coors Field": 115, "Great American Ball Park": 105,
    "Citizens Bank Park": 104, "Fenway Park": 104,
    "Globe Life Field": 103, "Globe Life Park in Arlington": 106,
    "Guaranteed Rate Field": 103, "Yankee Stadium": 103,
    "Wrigley Field": 103, "Kauffman Stadium": 99,
    "Angel Stadium": 97, "Petco Park": 96,
    "Oracle Park": 95, "T-Mobile Park": 97,
    "Tropicana Field": 97, "Dodger Stadium": 98,
    "loanDepot park": 97, "Marlins Park": 97,
    "PNC Park": 98, "Target Field": 100,
    "Progressive Field": 100, "Truist Park": 101,
    "SunTrust Park": 101, "Nationals Park": 101,
    "Minute Maid Park": 100, "Busch Stadium": 98,
    "American Family Field": 100, "Chase Field": 101,
    "Oriole Park at Camden Yards": 101, "Rogers Centre": 100,
    "Comerica Park": 97, "Oakland Coliseum": 96, "Citi Field": 98,
    "Daikin Park": 100, "Sutter Health Park": 96,
    "UNIQLO Field at Dodger Stadium": 98,
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", name)
        if not unicodedata.combining(c)
    ).lower().strip()


def get_today_games(game_date: str) -> list[dict]:
    """Fetch today's scheduled games with starting pitchers."""
    schedule = _retry(statsapi.schedule, start_date=game_date, end_date=game_date, sportId=1)
    games = [g for g in schedule if g["game_type"] == "R"]
    print(f"  {len(games)} regular season games on {game_date}")

    result = []
    for g in games:
        game_pk = g["game_id"]
        # Try to get confirmed SP from probable pitchers
        home_sp, away_sp = None, None
        try:
            details = _retry(statsapi.get, "game", {"gamePk": game_pk})
            probs = details.get("gameData", {}).get("probablePitchers", {})
            home_sp = probs.get("home", {}).get("fullName")
            away_sp = probs.get("away", {}).get("fullName")
        except Exception:
            pass

        result.append({
            "game_pk": game_pk,
            "date": game_date,
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "venue_name": g.get("venue_name"),
            "game_time_utc": g.get("game_datetime"),
            "home_sp_name": home_sp,
            "away_sp_name": away_sp,
            "status": g["status"],
        })
        time.sleep(0.1)

    return result


def get_recent_team_stats(game_date: str, n_games: int = 14) -> pd.DataFrame:
    """Pull last n_games for each team to compute rolling offense and RA."""
    from datetime import datetime, timedelta
    end_dt = datetime.strptime(game_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=60)  # look back 60 days to find n_games

    schedule = _retry(statsapi.schedule,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=(end_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        sportId=1,
    )
    games = [g for g in schedule if g["status"] == "Final" and g["game_type"] == "R"]

    rows = []
    for g in games:
        rows.append({
            "date": g["game_date"],
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "home_runs": g["home_score"],
            "away_runs": g["away_score"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Build per-team rolling stats
    records = {}
    for _, row in df.iterrows():
        for side, opp in [("home", "away"), ("away", "home")]:
            team = row[f"{side}_team"]
            runs_scored = row[f"{side}_runs"]
            runs_allowed = row[f"{opp}_runs"]
            if team not in records:
                records[team] = []
            records[team].append((row["date"], runs_scored, runs_allowed))

    team_stats = {}
    for team, game_list in records.items():
        game_list = sorted(game_list)[-n_games:]
        if len(game_list) >= 5:
            rpg = np.mean([r for _, r, _ in game_list])
            ra9 = np.mean([a for _, _, a in game_list])
        else:
            rpg, ra9 = np.nan, np.nan
        team_stats[team] = {"rolling_rpg": rpg, "rolling_ra9": ra9}

    return team_stats


def get_rest_days(game_date: str) -> dict:
    """Return days of rest for each team going into today's games."""
    from datetime import datetime, timedelta
    end_dt = datetime.strptime(game_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=10)

    schedule = _retry(statsapi.schedule,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=(end_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        sportId=1,
    )
    games = [g for g in schedule if g["status"] == "Final" and g["game_type"] == "R"]

    last_game = {}
    for g in sorted(games, key=lambda x: x["game_date"]):
        d = datetime.strptime(g["game_date"], "%Y-%m-%d")
        last_game[g["home_name"]] = d
        last_game[g["away_name"]] = d

    rest = {}
    today = datetime.strptime(game_date, "%Y-%m-%d")
    for team, last in last_game.items():
        days = (today - last).days
        rest[team] = min(days, 7)

    return rest


def get_forecast_weather(venue_name: str, game_date: str, game_hour_local: int = 19) -> dict:
    """Fetch weather forecast from Open-Meteo for the game's park."""
    if venue_name in DOMES:
        return {"temp_f": None, "wind_mph": 0.0, "wind_deg": None, "wind_out": False}

    park = PARK_INFO.get(venue_name)
    if not park:
        return {"temp_f": None, "wind_mph": None, "wind_deg": None, "wind_out": False}

    lat, lon, cf_deg = park
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
        f"&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=auto"
        f"&start_date={game_date}&end_date={game_date}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        idx = min(game_hour_local, len(hourly.get("temperature_2m", [])) - 1)
        temp = hourly.get("temperature_2m", [None])[idx]
        wind = hourly.get("windspeed_10m", [None])[idx]
        wdir = hourly.get("winddirection_10m", [None])[idx]
        wind_out = False
        if wdir is not None:
            diff = abs((wdir - cf_deg + 180) % 360 - 180)
            wind_out = diff <= 45
        return {"temp_f": temp, "wind_mph": wind, "wind_deg": wdir, "wind_out": wind_out}
    except Exception:
        return {"temp_f": None, "wind_mph": None, "wind_deg": None, "wind_out": False}


def load_sp_stats(season: int) -> pd.DataFrame:
    """Load current season pitcher stats."""
    path = RAW_DIR / f"pitching_stats_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Compute derived columns from BRef raw columns (mirrors build_features.py logic)
    ip = df["IP"].clip(lower=0.1)
    if "FIP" not in df.columns:
        df["FIP"] = ((13 * df["HR"] + 3 * df["BB"] - 2 * df["SO"]) / ip + 3.10).clip(1.0, 10.0)
    if "BB/9" not in df.columns:
        df["BB/9"] = (df["BB"] / ip * 9).clip(0, 15)
    if "K/9" not in df.columns:
        df["K/9"] = df["SO9"] if "SO9" in df.columns else (df["SO"] / ip * 9)
        df["K/9"] = df["K/9"].clip(0, 20)
    if "ip_per_gs" not in df.columns:
        df["ip_per_gs"] = (df["IP"] / df["GS"].clip(lower=1)).clip(0, 9)
    df["name_norm"] = df["Name"].apply(normalize_name)
    return df.set_index("name_norm")


def get_lines_oddsapi(game_date: str) -> dict:
    """
    Fetch today's MLB totals from The Odds API (free tier).
    Caches response to disk per date — re-runs on the same day use the cache
    and don't consume credits.
    Returns dict of {(home_team, away_team): total}
    Returns empty dict if no API key set.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return {}

    cache_dir = Path(__file__).parent.parent / "data" / "lines_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"odds_{game_date}.json"

    # Use cache if already fetched today
    if cache_path.exists():
        import json
        raw = json.loads(cache_path.read_text())
        lines = {tuple(k.split("|||")): v for k, v in raw.items()}
        print(f"  Odds API: {len(lines)} lines loaded from cache (no credit used)")
        return lines

    url = (
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
        f"?apiKey={api_key}&regions=us&markets=totals&oddsFormat=american"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        lines = {}
        for game in resp.json():
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == "Over":
                                # Store both orderings so matching works regardless of API convention
                                lines[(home, away)] = outcome["point"]
                                lines[(away, home)] = outcome["point"]
                                break
                        break

        # Save to cache (deduplicated)
        import json
        cache_path.write_text(json.dumps({"|||".join(k): v for k, v in lines.items()}))
        print(f"  Odds API: {len(lines)} lines fetched  ({remaining} credits remaining)")
        return lines
    except Exception as e:
        print(f"  Odds API error: {e}")
        return {}


def get_prior_season_lg_rpg(season: int) -> float:
    """Load prior season league average R/G from saved game data."""
    prior = season - 1
    path = RAW_DIR / f"games_{prior}.csv"
    if not path.exists():
        return 9.1  # fallback league average
    df = pd.read_csv(path)
    return df["total_runs"].mean()


def build_game_features(game: dict, sp_stats: pd.DataFrame, team_stats: dict,
                        rest: dict, season: int) -> dict:
    """Assemble the feature vector for one game."""
    venue = game.get("venue_name", "")
    wx = get_forecast_weather(venue, game["date"])

    home_sp_norm = normalize_name(game.get("home_sp_name") or "")
    away_sp_norm = normalize_name(game.get("away_sp_name") or "")

    def sp_stat(name_norm, col):
        if name_norm and name_norm in sp_stats.index:
            val = sp_stats.loc[name_norm, col]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            return float(val) if pd.notna(val) else np.nan
        return np.nan

    home_ts = team_stats.get(game["home_team"], {})
    away_ts = team_stats.get(game["away_team"], {})

    return {
        "home_sp_fip":      sp_stat(home_sp_norm, "FIP"),
        "home_sp_k_9":      sp_stat(home_sp_norm, "K/9"),
        "home_sp_bb_9":     sp_stat(home_sp_norm, "BB/9"),
        "home_sp_ip_per_gs":sp_stat(home_sp_norm, "ip_per_gs"),
        "home_sp_era":      sp_stat(home_sp_norm, "ERA"),
        "away_sp_fip":      sp_stat(away_sp_norm, "FIP"),
        "away_sp_k_9":      sp_stat(away_sp_norm, "K/9"),
        "away_sp_bb_9":     sp_stat(away_sp_norm, "BB/9"),
        "away_sp_ip_per_gs":sp_stat(away_sp_norm, "ip_per_gs"),
        "away_sp_era":      sp_stat(away_sp_norm, "ERA"),
        "combined_sp_fip":  sp_stat(home_sp_norm, "FIP") + sp_stat(away_sp_norm, "FIP"),
        "home_rolling_rpg": home_ts.get("rolling_rpg", np.nan),
        "away_rolling_rpg": away_ts.get("rolling_rpg", np.nan),
        "home_rolling_ra9": home_ts.get("rolling_ra9", np.nan),
        "away_rolling_ra9": away_ts.get("rolling_ra9", np.nan),
        "park_factor":      PARK_FACTORS.get(venue, 100),
        "temp_f":           wx["temp_f"],
        "wind_mph":         wx["wind_mph"],
        "wind_out_num":     float(wx["wind_out"]),
        "home_rest_days":   rest.get(game["home_team"], 3),
        "away_rest_days":   rest.get(game["away_team"], 3),
        "lg_rpg":           get_prior_season_lg_rpg(season),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--edge", type=float, default=MIN_EDGE)
    args = parser.parse_args()
    game_date = args.date
    season = int(game_date[:4])

    print(f"\n{'='*55}")
    print(f"  MLB Totals Model — {game_date}")
    print(f"{'='*55}")

    # Load model
    model_path = MODELS_DIR / "ridge_production.joblib"
    meta_path = MODELS_DIR / "ridge_meta.joblib"
    if not model_path.exists():
        print("ERROR: No production model found. Run: python src/train.py")
        return
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    feature_cols = meta["feature_cols"]
    bias = meta["bias"]
    print(f"  Model loaded: {len(feature_cols)} features, bias={bias:+.3f}")

    # Today's games
    print("\nFetching today's schedule...")
    games = get_today_games(game_date)
    if not games:
        print("No games today.")
        return

    # Rolling team stats
    print("Fetching recent team stats...")
    team_stats = get_recent_team_stats(game_date)

    # Rest days
    rest = get_rest_days(game_date)

    # SP stats — use current season if available, otherwise fall back to most recent
    print(f"Loading {season} pitcher stats...")
    sp_stats = load_sp_stats(season)
    if sp_stats.empty:
        for fallback in range(season - 1, season - 4, -1):
            sp_stats = load_sp_stats(fallback)
            if not sp_stats.empty:
                print(f"  No {season} stats yet — using {fallback} as fallback.")
                break
        if sp_stats.empty:
            print(f"  Warning: no pitcher stats found. Run fetch_stats.py.")

    # Lines (optional)
    print("Fetching current lines...")
    lines = get_lines_oddsapi(game_date)
    if not lines:
        print("  No Odds API key set — predictions shown without lines.")
        print("  Set ODDS_API_KEY env var to include current lines.")

    # Build features and predict
    rows = []
    for game in games:
        feats = build_game_features(game, sp_stats, team_stats, rest, season)
        X = pd.DataFrame([feats])[feature_cols]
        raw_pred = float(model.predict(X)[0])
        predicted = raw_pred - bias

        # Match line from Odds API — normalize both sides for fuzzy matching
        line = None
        home_norm = normalize_name(game["home_team"])
        away_norm = normalize_name(game["away_team"])
        for (h, a), total in lines.items():
            h_norm = normalize_name(h)
            a_norm = normalize_name(a)
            # Match on any word overlap (handles "Athletics" vs "Oakland Athletics" etc.)
            home_words = set(home_norm.split())
            away_words = set(away_norm.split())
            if home_words & set(h_norm.split()) and away_words & set(a_norm.split()):
                line = total
                break

        edge = (predicted - line) if line is not None else None
        bet = None
        watch = False
        if edge is not None and not np.isnan(edge):
            if args.edge <= abs(edge) <= MAX_EDGE:
                bet = "OVER" if edge > 0 else "UNDER"
            elif abs(edge) > MAX_EDGE:
                watch = True  # exceeds cap — track but don't bet

        rows.append({
            "away": game["away_team"],
            "home": game["home_team"],
            "away_sp": game.get("away_sp_name") or "TBD",
            "home_sp": game.get("home_sp_name") or "TBD",
            "predicted": round(predicted, 2),
            "line": line,
            "edge": round(edge, 2) if edge is not None else None,
            "bet": bet,
            "watch": watch,
            "temp_f": feats.get("temp_f"),
            "wind_mph": feats.get("wind_mph"),
            "wind_out": bool(feats.get("wind_out_num")),
        })

    df = pd.DataFrame(rows)

    # Print full game list
    print(f"\n{'─'*55}")
    print(f"  {'AWAY':<22} {'HOME':<22} {'PRED':>5} {'LINE':>5} {'EDGE':>6} {'BET':>6}")
    print(f"{'─'*55}")
    for _, r in df.iterrows():
        away = r["away"][:20]
        home = r["home"][:20]
        pred = f"{r['predicted']:.1f}"
        line_val = r["line"]
        edge_val = r["edge"]
        line_str = f"{line_val:.1f}" if pd.notna(line_val) else "  —"
        edge_str = f"{edge_val:+.2f}" if pd.notna(edge_val) else "  —"
        bet_str = r["bet"] if pd.notna(r["bet"]) else ""
        flag = " ◄" if bet_str else ""
        print(f"  {away:<22} {home:<22} {pred:>5} {line_str:>5} {edge_str:>6} {bet_str:>6}{flag}")

    bets = df[df["bet"].notna()]
    watch_bets = df[df["watch"] == True]
    print(f"\n  {len(bets)} bet(s) flagged (edge {args.edge}–{MAX_EDGE} runs)")

    if not bets.empty:
        print(f"\n{'─'*55}")
        print("  FLAGGED BETS:")
        for _, r in bets.iterrows():
            print(f"  {r['bet']} {r['away']} @ {r['home']}")
            print(f"    Predicted: {r['predicted']:.2f}  Line: {r['line']}  Edge: {r['edge']:+.2f}")
            print(f"    {r['away_sp']} vs {r['home_sp']}")
            if r["temp_f"]:
                wind_note = " (wind out)" if r["wind_out"] else ""
                print(f"    Weather: {r['temp_f']:.0f}°F, {r['wind_mph']:.0f} mph{wind_note}")

    if not watch_bets.empty:
        print(f"\n{'─'*55}")
        print(f"  WATCH LIST (edge >{MAX_EDGE} — not betting):")
        for _, r in watch_bets.iterrows():
            direction = "OVER" if r["edge"] > 0 else "UNDER"
            print(f"  {direction} {r['away']} @ {r['home']}"
                  f"  |  Pred: {r['predicted']:.1f}  Line: {r['line']}  Edge: {r['edge']:+.2f}")

    # Save output
    out_dir = Path(__file__).parent.parent / "data" / "predictions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"predictions_{game_date}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path.name}")

    # Return flagged bets and watch list for daily_run / notify
    def _to_bet_dict(r, bet_label=None):
        wx_parts = []
        if pd.notna(r.get("temp_f")):
            note = " (wind out)" if r.get("wind_out") else ""
            wx_parts.append(f"{r['temp_f']:.0f}°F, {r['wind_mph']:.0f} mph{note}")
        return {
            "bet":       bet_label or r.get("bet", ""),
            "away":      r["away"],
            "home":      r["home"],
            "predicted": r["predicted"],
            "line":      r["line"],
            "edge":      r["edge"],
            "away_sp":   r.get("away_sp", "TBD"),
            "home_sp":   r.get("home_sp", "TBD"),
            "weather":   wx_parts[0] if wx_parts else "",
        }

    flagged = [_to_bet_dict(r) for _, r in bets.iterrows()]
    watch_list = [_to_bet_dict(r, "OVER" if r["edge"] > 0 else "UNDER")
                  for _, r in watch_bets.iterrows()]
    return flagged, watch_list


if __name__ == "__main__":
    main()
