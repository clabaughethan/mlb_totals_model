"""
Fill in actual results for past prediction CSVs and print running P&L.

Run the morning after games complete:
  python src/update_results.py

Scans data/predictions/ for any CSVs missing actual_total, pulls final
scores from statsapi, fills in results, and prints a cumulative summary.
"""

import sys
import time
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import numpy as np
import statsapi

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRED_DIR = Path(__file__).parent.parent / "data" / "predictions"
MIN_EDGE = 0.75
MAX_EDGE = 1.5

UNIT_SIZES = [10, 20, 25, 50, 100]


def fetch_results(game_date: str) -> dict:
    """Return {(away_team, home_team): total_runs} for all final games on date."""
    try:
        schedule = statsapi.schedule(start_date=game_date, end_date=game_date, sportId=1)
    except Exception as e:
        print(f"  statsapi error for {game_date}: {e}")
        return {}

    results = {}
    for g in schedule:
        if g["status"] == "Final" and g["game_type"] == "R":
            total = g["home_score"] + g["away_score"]
            results[(g["away_name"], g["home_name"])] = total
    return results


def fill_results(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Add actual_total, won, push, profit_units to rows that have a bet and no result yet."""
    df = df.copy()
    expected_dtypes = {
        "actual_total": "float64",
        "won": "boolean",
        "push": "boolean",
        "profit_units": "float64",
    }

    for col, dtype in expected_dtypes.items():
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype=dtype)
        else:
            if dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            else:
                df[col] = df[col].astype(dtype)

    for idx, row in df.iterrows():
        if pd.notna(row.get("actual_total")):
            continue

        key = (row["away"], row["home"])
        if key not in results:
            continue

        actual = results[key]
        df.at[idx, "actual_total"] = actual

        if pd.isna(row.get("bet")) or row.get("bet") == "":
            continue

        line = row.get("line")
        if pd.isna(line):
            continue

        bet = row["bet"]
        push = actual == line
        won = (actual > line) if bet == "OVER" else (actual < line)

        df.at[idx, "push"] = push
        df.at[idx, "won"] = won if not push else False
        df.at[idx, "profit_units"] = 0.0 if push else (1.0 if won else -1.1)

    return df


def score_block(scored: pd.DataFrame, label: str) -> dict:
    """Compute summary stats for a slice of scored bets. Returns dict for reuse in email."""
    if scored.empty:
        return None

    n = len(scored)
    wins = int(scored["won"].sum())
    pushes = int(scored["push"].sum())
    non_push = n - pushes
    losses = non_push - wins
    profit = scored["profit_units"].sum()
    roi = profit / (n * 1.1) * 100
    win_rate = wins / non_push if non_push > 0 else 0

    return {
        "label": label,
        "n": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "profit_units": profit,
        "roi": roi,
    }


def print_block(stats: dict):
    if not stats:
        print("  No scored bets.")
        return
    w, l, p = stats["wins"], stats["losses"], stats["pushes"]
    print(f"  Record:     {w}-{l}-{p}  ({stats['win_rate']:.1%} win rate)")
    print(f"  Units:      {stats['profit_units']:+.2f}")
    print(f"  ROI:        {stats['roi']:+.1f}%  (break-even: 52.4%)")
    for unit in UNIT_SIZES:
        dollar = stats["profit_units"] * unit
        print(f"  ${unit:>3}/bet:   ${dollar:+.2f}")


def build_summary(all_bets: pd.DataFrame) -> dict:
    """Build full summary dict: overall, L30, L7, by direction, by edge bucket."""
    if "profit_units" not in all_bets.columns:
        return {}

    scored = all_bets.dropna(subset=["profit_units"]).copy()
    scored["date"] = pd.to_datetime(scored["date"])
    scored = scored.sort_values("date")

    if scored.empty:
        return {}

    today = scored["date"].max()
    l30_cutoff = today - timedelta(days=30)
    l7_cutoff = today - timedelta(days=7)

    overall = score_block(scored, "Overall")
    l30 = score_block(scored[scored["date"] >= l30_cutoff], "Last 30 days")
    l7 = score_block(scored[scored["date"] >= l7_cutoff], "Last 7 days")

    # By direction
    by_dir = {}
    for direction in ["OVER", "UNDER"]:
        subset = scored[scored["bet"] == direction]
        by_dir[direction] = score_block(subset, direction)

    # By edge bucket
    by_edge = {}
    if "edge" in scored.columns:
        scored["edge_bucket"] = pd.cut(
            scored["edge"].abs(),
            bins=[0, 1.0, 1.5, 99],
            labels=["0.75-1.0", "1.0-1.5", "1.5+"],
        )
        for bucket in ["0.75-1.0", "1.0-1.5", "1.5+"]:
            subset = scored[scored["edge_bucket"] == bucket]
            by_edge[bucket] = score_block(subset, f"Edge {bucket}")

    # Recent bets table (last 10)
    recent_cols = ["date", "away", "home", "bet", "line", "edge", "actual_total", "won", "profit_units"]
    recent_cols = [c for c in recent_cols if c in scored.columns]
    recent = scored.tail(10)[recent_cols].copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")

    return {
        "overall": overall,
        "l30": l30,
        "l7": l7,
        "by_direction": by_dir,
        "by_edge": by_edge,
        "recent": recent,
    }


def print_summary(all_bets: pd.DataFrame):
    summary = build_summary(all_bets)
    if not summary:
        print("  No completed bets yet.")
        return

    for label, stats in [("OVERALL", summary["overall"]),
                          ("LAST 30 DAYS", summary["l30"]),
                          ("LAST 7 DAYS", summary["l7"])]:
        print(f"\n  {'─'*40}")
        print(f"  {label}")
        print(f"  {'─'*40}")
        print_block(stats)

    print(f"\n  {'─'*40}")
    print(f"  BY DIRECTION")
    print(f"  {'─'*40}")
    for direction, stats in summary["by_direction"].items():
        if stats:
            print(f"\n  {direction}:")
            print_block(stats)

    print(f"\n  {'─'*40}")
    print(f"  BY EDGE BUCKET")
    print(f"  {'─'*40}")
    for bucket, stats in summary["by_edge"].items():
        if stats:
            print(f"\n  Edge {bucket}:")
            print_block(stats)

    print(f"\n  {'─'*40}")
    print(f"  RECENT BETS (last 10)")
    print(f"  {'─'*40}")
    print(summary["recent"].to_string(index=False))


def main():
    PRED_DIR.mkdir(exist_ok=True)
    pred_files = sorted(PRED_DIR.glob("predictions_*.csv"))

    if not pred_files:
        print("No prediction files found in data/predictions/")
        return

    print(f"Found {len(pred_files)} prediction file(s).\n")

    all_bets = []
    updated = 0

    for path in pred_files:
        date_str = path.stem.replace("predictions_", "")
        df = pd.read_csv(path)

        needs_update = (
            df["bet"].notna() &
            (df["bet"] != "") &
            df.get("actual_total", pd.Series([np.nan] * len(df))).isna()
        )

        if needs_update.any():
            print(f"Fetching results for {date_str}...")
            results = fetch_results(date_str)

            if results:
                df = fill_results(df, results)
                df.to_csv(path, index=False)
                filled = df["actual_total"].notna().sum()
                print(f"  {filled}/{len(df)} games filled.")
                updated += 1
            else:
                print(f"  No final scores yet for {date_str} — skipping.")
        else:
            print(f"{date_str}: already complete.")

        bets = df[df["bet"].notna() & (df["bet"] != "")].copy()
        if not bets.empty:
            bets["date"] = date_str
            all_bets.append(bets)

        time.sleep(0.1)

    if updated:
        print(f"\n{updated} file(s) updated.")

    if all_bets:
        combined = pd.concat(all_bets, ignore_index=True)
        print(f"\n{'='*45}")
        print("  LIVE TRACKING SUMMARY")
        print(f"{'='*45}")
        print_summary(combined)

        tracker_path = PRED_DIR / "tracker.csv"
        combined.to_csv(tracker_path, index=False)
        print(f"\n  Full tracker saved: {tracker_path.name}")

        summary = build_summary(combined)

        # Pull yesterday's scored bets for the email
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        if "profit_units" not in combined.columns:
            combined["profit_units"] = np.nan
        scored = combined[
            combined["date"].astype(str).str.startswith(yesterday) &
            combined["bet"].notna() &
            combined["profit_units"].notna()
        ]

        yesterday_results = [
            {
                "bet":          r["bet"],
                "away":         r["away"],
                "home":         r["home"],
                "line":         r["line"],
                "actual":       r["actual_total"],
                "won":          bool(r["won"]),
                "push":         bool(r["push"]),
                "profit_units": r["profit_units"],
            }
            for _, r in scored.iterrows()
        ]

        return summary, yesterday_results
    else:
        print("\nNo bets placed yet.")
        return {}, []


if __name__ == "__main__":
    main()
