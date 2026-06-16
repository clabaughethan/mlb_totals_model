"""
Fetch season-level pitcher and team batting stats via pybaseball (FanGraphs).
Outputs:
  data/raw/pitching_stats_{year}.csv  -- SP stats (min 10 GS)
  data/raw/team_batting_{year}.csv    -- team offense

Run once; results are cached to disk.
"""

import pybaseball as pb
import pandas as pd
from pathlib import Path
import unicodedata


def normalize_pitcher_name(s: str) -> str:
    """
    Pybaseball returns accented names with literal backslash-x escape sequences
    (e.g. 'Rodr\\xc3\\xadguez' — 4 ASCII chars per encoded byte).
    Convert each \\xNN sequence to its byte value, decode the resulting bytes as UTF-8,
    then strip combining characters to produce plain ASCII.
    """
    if not isinstance(s, str):
        return s
    import re
    # Replace literal \xNN sequences with actual bytes
    try:
        byte_str = re.sub(
            r"\\x([0-9a-fA-F]{2})",
            lambda m: bytes.fromhex(m.group(1)).decode("latin-1"),
            s,
        )
        # Now byte_str has Latin-1 chars; re-encode and decode as UTF-8
        fixed = byte_str.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        fixed = s
    return "".join(
        c for c in unicodedata.normalize("NFKD", fixed)
        if not unicodedata.combining(c)
    )

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = list(range(2015, 2027))

# Suppress pybaseball progress bars
pb.cache.enable()


def fetch_pitching(year: int) -> pd.DataFrame:
    out_path = RAW_DIR / f"pitching_stats_{year}.csv"
    if out_path.exists():
        print(f"{year} pitching: already fetched.")
        return pd.read_csv(out_path)

    print(f"Fetching {year} pitching stats...")
    # Use Baseball Reference scraper — FanGraphs legacy endpoint returns 403
    df = pb.pitching_stats_bref(year)
    # BRef uses 'G' and 'GS' columns
    if "GS" not in df.columns and "G" in df.columns:
        df["GS"] = df["G"]
    df = df[df["GS"] >= 1].copy()  # any pitcher with at least one start
    # Fix mojibake from pybaseball and strip accents — ensures plain ASCII names in CSV
    if "Name" in df.columns:
        df["Name"] = df["Name"].apply(normalize_pitcher_name)
    rename = {"ERA+": "ERA_plus", "SO/W": "k_bb_ratio"}
    df = df.rename(columns=rename)
    df["season"] = year
    df.to_csv(out_path, index=False)
    print(f"  {len(df)} pitchers saved.")
    return df


def fetch_team_batting(year: int) -> pd.DataFrame:
    out_path = RAW_DIR / f"team_batting_{year}.csv"
    if out_path.exists():
        print(f"{year} team batting: already fetched.")
        return pd.read_csv(out_path)

    print(f"Fetching {year} team batting stats...")
    # Aggregate player-level BRef batting to team totals
    df = pb.batting_stats_bref(year)
    # Keep only rows with a real team (not league totals)
    df = df[df["Tm"].notna() & ~df["Tm"].isin(["", "TOT", "AL", "NL"])]
    agg = df.groupby("Tm").agg(
        R=("R", "sum"),
        PA=("PA", "sum"),
        H=("H", "sum"),
        HR=("HR", "sum"),
        BB=("BB", "sum"),
        SO=("SO", "sum"),
        G=("G", "max"),
    ).reset_index()
    agg["R_per_G"] = agg["R"] / agg["G"].clip(lower=1)
    agg["season"] = year
    agg.to_csv(out_path, index=False)
    print(f"  {len(agg)} teams saved.")
    return agg


def main():
    for year in SEASONS:
        fetch_pitching(year)
        fetch_team_batting(year)

    print("\nAll stats fetched.")

    # Quick preview
    df = pd.read_csv(RAW_DIR / "pitching_stats_2023.csv")
    print("\n2023 pitching sample:")
    cols = ["Name", "Tm", "GS", "IP", "ERA", "FIP", "xFIP", "SO9", "BB9"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values("GS", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
