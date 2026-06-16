"""
Fetch historical hourly weather for each MLB game via Open-Meteo (free, no key).
Looks up temperature and wind at game time for each park.

Outputs: data/raw/weather.csv

Columns: game_pk, date, venue_name, temp_f, wind_mph, wind_deg, wind_out (bool estimate)
"""

import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Park coordinates (lat, lon) and approximate wind-out direction in degrees
# Wind "blows out" when it matches the orientation of the field toward CF
PARK_INFO = {
    "Oriole Park at Camden Yards":       (39.2839, -76.6218, 230),
    "Fenway Park":                        (42.3467, -71.0972, 270),
    "Yankee Stadium":                     (40.8296, -73.9262, 0),
    "Rogers Centre":                      (43.6414, -79.3894, 0),
    "Guaranteed Rate Field":              (41.8300, -87.6338, 135),
    "Wrigley Field":                      (41.9484, -87.6553, 315),
    "Progressive Field":                  (41.4962, -81.6852, 225),
    "Comerica Park":                      (42.3390, -83.0485, 335),
    "Kauffman Stadium":                   (39.0517, -94.4803, 0),
    "Target Field":                       (44.9817, -93.2781, 0),
    "Tropicana Field":                    (27.7682, -82.6534, 0),   # dome, wind N/A
    "Globe Life Field":                   (32.7473, -97.0824, 0),   # retractable
    "Minute Maid Park":                   (29.7573, -95.3555, 0),   # retractable
    "Oakland Coliseum":                   (37.7516, -122.2005, 270),
    "T-Mobile Park":                      (47.5914, -122.3325, 315),
    "Angel Stadium":                      (33.8003, -117.8827, 225),
    "Dodger Stadium":                     (34.0739, -118.2400, 315),
    "Petco Park":                         (32.7076, -117.1570, 270),
    "Oracle Park":                        (37.7786, -122.3893, 270),
    "Coors Field":                        (39.7559, -104.9942, 315),
    "Chase Field":                        (33.4455, -112.0667, 0),   # retractable
    "Busch Stadium":                      (38.6226, -90.1928, 315),
    "American Family Field":              (43.0280, -87.9712, 0),
    "PNC Park":                           (40.4469, -80.0057, 135),
    "Great American Ball Park":           (39.0979, -84.5082, 225),
    "Truist Park":                        (33.8908, -84.4678, 0),
    "Nationals Park":                     (38.8730, -77.0074, 270),
    "loanDepot park":                     (25.7781, -80.2197, 0),   # retractable
    "Citi Field":                         (40.7571, -73.8458, 0),
    "Citizens Bank Park":                 (39.9061, -75.1665, 180),
    # Older/renamed
    "Globe Life Park in Arlington":       (32.7512, -97.0832, 0),
    "SunTrust Park":                      (33.8908, -84.4678, 0),
    "Marlins Park":                       (25.7781, -80.2197, 0),
    "Daikin Park":                        (29.7573, -95.3555, 0),   # fka Minute Maid Park, retractable
    "Sutter Health Park":                 (38.5727, -121.4944, 315), # Athletics temp home, Sacramento
    "UNIQLO Field at Dodger Stadium":     (34.0739, -118.2400, 315),
}

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={date}&end_date={date}"
    "&hourly=temperature_2m,windspeed_10m,winddirection_10m"
    "&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=auto"
)

DOMES = {"Tropicana Field", "Chase Field", "Minute Maid Park", "Daikin Park",
         "Globe Life Field", "loanDepot park", "Marlins Park", "Rogers Centre"}


def wind_is_out(wind_deg: float, park_cf_deg: float, threshold: float = 45.0) -> bool:
    """True if wind is blowing roughly toward CF (within threshold degrees)."""
    diff = abs((wind_deg - park_cf_deg + 180) % 360 - 180)
    return diff <= threshold


def fetch_game_weather(lat: float, lon: float, date: str, game_hour_utc: int = 19) -> dict:
    url = OPEN_METEO_URL.format(lat=lat, lon=lon, date=date)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("windspeed_10m", [])
        wind_dirs = hourly.get("winddirection_10m", [])

        # Pick the hour closest to first pitch (default 7pm local)
        idx = min(game_hour_utc, len(temps) - 1) if temps else 0
        return {
            "temp_f": temps[idx] if temps else None,
            "wind_mph": winds[idx] if winds else None,
            "wind_deg": wind_dirs[idx] if wind_dirs else None,
        }
    except Exception:
        return {"temp_f": None, "wind_mph": None, "wind_deg": None}


WORKERS = 20
BATCH_SIZE = 500
_write_lock = threading.Lock()


def process_row(row):
    venue = row["venue_name"]
    park = PARK_INFO.get(venue)
    is_dome = venue in DOMES

    if is_dome or park is None:
        return {
            "game_pk": row["game_pk"],
            "date": row["date"],
            "venue_name": venue,
            "temp_f": None,
            "wind_mph": 0.0 if is_dome else None,
            "wind_deg": None,
            "wind_out": False,
            "is_dome": is_dome,
        }

    lat, lon, cf_deg = park
    wx = fetch_game_weather(lat, lon, row["date"])
    wind_out = wind_is_out(wx["wind_deg"], cf_deg) if wx["wind_deg"] is not None else False

    return {
        "game_pk": row["game_pk"],
        "date": row["date"],
        "venue_name": venue,
        **wx,
        "wind_out": wind_out,
        "is_dome": False,
    }


def flush_batch(out_path, rows):
    with _write_lock:
        new_df = pd.DataFrame(rows)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(out_path, index=False)


def main():
    out_path = RAW_DIR / "weather.csv"

    games_path = RAW_DIR / "games_all.csv"
    if games_path.exists():
        games = pd.read_csv(games_path, usecols=["game_pk", "date", "venue_name", "game_time_utc"])
    else:
        season_files = sorted(RAW_DIR.glob("games_20??.csv"))
        if not season_files:
            print("No game data found — run fetch_data.py first.")
            return
        games = pd.concat(
            [pd.read_csv(p, usecols=["game_pk", "date", "venue_name", "game_time_utc"]) for p in season_files],
            ignore_index=True,
        )
        print(f"  (games_all.csv not ready — loaded {len(season_files)} season files)")
    games = games.dropna(subset=["venue_name"])

    if out_path.exists():
        done = set(pd.read_csv(out_path)["game_pk"].tolist())
        games = games[~games["game_pk"].isin(done)]
        print(f"{len(done)} games already fetched, {len(games)} remaining.")

    game_rows = [row for _, row in games.iterrows()]
    batch = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_row, row): row for row in game_rows}
        with tqdm(total=len(game_rows), desc="Weather") as pbar:
            for future in as_completed(futures):
                result = future.result()
                batch.append(result)
                pbar.update(1)
                if len(batch) >= BATCH_SIZE:
                    flush_batch(out_path, batch)
                    batch = []

    if batch:
        flush_batch(out_path, batch)

    total = len(pd.read_csv(out_path))
    print(f"\nWeather saved: {total} total games -> {out_path.name}")


if __name__ == "__main__":
    main()
