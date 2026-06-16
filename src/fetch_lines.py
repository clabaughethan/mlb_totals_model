"""
Load historical MLB betting lines from Sports Book Reviews Online (SBR).

SBR blocks automated downloads, so this script supports two modes:

MODE 1 (manual — recommended):
  1. Go to https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb-odds/
  2. Download each season's Excel file manually into data/raw/sbr/
     e.g. data/raw/sbr/mlb_odds_2018.xlsx, mlb_odds_2019.xlsx, ...
  3. Run: python src/fetch_lines.py

MODE 2 (auto attempt):
  The script will try direct download first. If SBR blocks it (returns HTML),
  it prints instructions for manual download and skips that year.

Outputs: data/raw/lines_{year}.csv and data/raw/lines_all.csv
"""

import requests
import pandas as pd
from pathlib import Path
from io import BytesIO
import glob

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SBR_DIR = RAW_DIR / "sbr"
SBR_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = list(range(2015, 2027))

SBR_URLS = [
    "https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb%20odds%20{year}.xlsx",
    "https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb-odds-{year}.xlsx",
]

TEAM_NAME_MAP = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC":  "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD":  "San Diego Padres",
    "SF":  "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB":  "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
    "CLV": "Cleveland Guardians",
    "CUB": "Chicago Cubs",
    "SDG": "San Diego Padres",
    "SFO": "San Francisco Giants",
    "TAM": "Tampa Bay Rays",
    "KAN": "Kansas City Royals",
    "CHW": "Chicago White Sox",
    "ANA": "Los Angeles Angels",
    "FLA": "Miami Marlins",
    "WAS": "Washington Nationals",
    "BRS": "Boston Red Sox",
    "KCR": "Kansas City Royals",
    "TBR": "Tampa Bay Rays",
    "SDP": "San Diego Padres",
}


def is_html(content: bytes) -> bool:
    return content[:100].lower().lstrip().startswith(b"<!doc")


def try_download(year: int) -> bytes | None:
    for url in SBR_URLS:
        try:
            resp = requests.get(
                url.format(year=year), timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            if resp.status_code == 200 and not is_html(resp.content):
                return resp.content
        except requests.RequestException:
            continue
    return None


def find_manual_file(year: int) -> Path | None:
    patterns = [
        SBR_DIR / f"*{year}*.xlsx",
        SBR_DIR / f"*{year}*.xls",
        RAW_DIR / f"*{year}*.xlsx",
        RAW_DIR / f"*{year}*.xls",
    ]
    for pat in patterns:
        matches = list(pat.parent.glob(pat.name))
        if matches:
            return matches[0]
    return None


def parse_sbr_excel(data: bytes | Path, year: int) -> pd.DataFrame:
    if isinstance(data, Path):
        raw = data.read_bytes()
    else:
        raw = data

    for engine in ("openpyxl", "xlrd"):
        try:
            df = pd.read_excel(BytesIO(raw), header=0, engine=engine)
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse Excel file for {year}")

    df.columns = [str(c).strip() for c in df.columns]

    rows = []
    i = 0
    while i < len(df) - 1:
        away_row = df.iloc[i]
        home_row = df.iloc[i + 1]

        if str(away_row.get("VH", "")).strip() != "V" or str(home_row.get("VH", "")).strip() != "H":
            i += 1
            continue

        raw_date = str(away_row.get("Date", "")).strip().zfill(4)
        try:
            month, day = int(raw_date[:2]), int(raw_date[2:])
            date_str = f"{year}-{month:02d}-{day:02d}"
        except Exception:
            i += 2
            continue

        away_abbr = str(away_row.get("Team", "")).strip()
        home_abbr = str(home_row.get("Team", "")).strip()

        def parse_num(val):
            try:
                return float(str(val).replace("½", ".5").replace("u", "").replace("o", ""))
            except (ValueError, TypeError):
                return None

        open_total = parse_num(away_row.get("Open OU") or away_row.get("OpenOU"))
        close_total = parse_num(away_row.get("Close OU") or away_row.get("CloseOU"))

        # Sanity check: totals should be in range 5-20
        if open_total and not (5 <= open_total <= 20):
            open_total = None
        if close_total and not (5 <= close_total <= 20):
            close_total = None

        rows.append({
            "date": date_str,
            "away_team_abbr": away_abbr,
            "home_team_abbr": home_abbr,
            "away_team": TEAM_NAME_MAP.get(away_abbr, away_abbr),
            "home_team": TEAM_NAME_MAP.get(home_abbr, home_abbr),
            "open_total": open_total,
            "close_total": close_total,
            "away_final": away_row.get("Final", away_row.get("F")),
            "home_final": home_row.get("Final", home_row.get("F")),
        })
        i += 2

    return pd.DataFrame(rows)


def fetch_lines(year: int) -> pd.DataFrame:
    out_path = RAW_DIR / f"lines_{year}.csv"
    if out_path.exists():
        print(f"{year} lines: already fetched.")
        return pd.read_csv(out_path)

    # Try manual file first
    manual = find_manual_file(year)
    if manual:
        print(f"{year}: found manual file {manual.name}, parsing...")
        df = parse_sbr_excel(manual, year)
        df["season"] = year
        df.to_csv(out_path, index=False)
        print(f"  {len(df)} games saved.")
        return df

    # Try auto download
    print(f"{year}: attempting download from SBR...")
    content = try_download(year)
    if content:
        df = parse_sbr_excel(content, year)
        df["season"] = year
        df.to_csv(out_path, index=False)
        print(f"  {len(df)} games saved.")
        return df

    print(f"  {year}: SBR download blocked. Manual download instructions:")
    print(f"    1. Visit: https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb-odds/")
    print(f"    2. Download the {year} Excel file")
    print(f"    3. Save to: {SBR_DIR}/mlb_odds_{year}.xlsx")
    print(f"    4. Re-run this script")
    return pd.DataFrame()


def main():
    all_lines = []
    missing = []
    for year in SEASONS:
        df = fetch_lines(year)
        if not df.empty:
            all_lines.append(df)
        else:
            missing.append(year)

    if all_lines:
        combined = pd.concat(all_lines, ignore_index=True)
        combined.to_csv(RAW_DIR / "lines_all.csv", index=False)
        print(f"\nCombined: {len(combined)} game lines saved.")
        sample_cols = ["date", "away_team", "home_team", "open_total", "close_total"]
        print(combined[sample_cols].dropna().head(10).to_string())

    if missing:
        print(f"\nMissing seasons (need manual download): {missing}")
        print(f"Save files to: {SBR_DIR}/")


if __name__ == "__main__":
    main()
