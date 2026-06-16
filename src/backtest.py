"""
Backtest the model's betting performance against historical closing lines.

Key metrics:
  - ROI: (profit / total wagered) * 100
  - CLV: average (predicted_total - close_total) in the direction of our bet
    Positive CLV = we consistently bet before the line moved against us
  - Win rate, units won/lost at -110 vig

Usage:
  python src/backtest.py --model ridge --edge 1.0 --min-edge 0.5
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / "predictions.csv"
    return pd.read_csv(path, parse_dates=["date"])


def simulate_bet(row: pd.Series, model_col: str, edge_threshold: float, max_edge: float = None) -> dict | None:
    """
    Decide whether to bet Over or Under based on model vs. close line.
    Returns bet dict or None if no bet.

    edge_threshold: minimum abs(predicted - line) to place a bet
    max_edge: maximum abs(predicted - line) — bets beyond this are skipped

    CLV (Closing Line Value): measures whether the open line moved in our favor by close.
      - We simulate betting at the open line.
      - If we bet Over and close > open: positive CLV (line moved against bettors = we got value).
      - If we bet Under and close < open: positive CLV.
      - NaN if open_total is missing.
    """
    predicted = row[model_col]
    line = row["close_total"]
    actual = row["actual_total"]
    open_line = row.get("open_total", np.nan)

    if pd.isna(predicted) or pd.isna(line) or pd.isna(actual):
        return None

    diff = predicted - line
    if abs(diff) < edge_threshold:
        return None
    if max_edge is not None and abs(diff) > max_edge:
        return None

    bet_over = diff > 0
    won = (actual > line) if bet_over else (actual < line)
    push = actual == line

    if push:
        profit = 0.0
    elif won:
        profit = 1.0
    else:
        profit = -1.1

    # True CLV: line movement from open to close in the direction of our bet
    if pd.notna(open_line):
        line_move = line - open_line  # positive = line went up
        clv = line_move if bet_over else -line_move
    else:
        clv = np.nan

    return {
        "game_pk": row["game_pk"],
        "date": row["date"],
        "season": row["season"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "predicted": predicted,
        "open_total": open_line,
        "close_total": line,
        "actual_total": actual,
        "edge": diff,
        "bet": "OVER" if bet_over else "UNDER",
        "won": won,
        "push": push,
        "profit_units": profit,
        "clv": clv,
    }


def run_backtest(preds: pd.DataFrame, model_col: str, edge_threshold: float, max_edge: float = None) -> pd.DataFrame:
    bets = []
    for _, row in preds.iterrows():
        bet = simulate_bet(row, model_col, edge_threshold, max_edge)
        if bet:
            bets.append(bet)
    return pd.DataFrame(bets)


def print_summary(bets: pd.DataFrame, label: str):
    if bets.empty:
        print(f"\n{label}: No bets placed.")
        return

    n = len(bets)
    wins = bets["won"].sum()
    pushes = bets["push"].sum()
    total_profit = bets["profit_units"].sum()
    roi = (total_profit / (n * 1.1)) * 100
    win_rate = wins / (n - pushes) if (n - pushes) > 0 else 0
    avg_clv = bets["clv"].mean()

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"  Bets placed:    {n}")
    print(f"  Win rate:       {win_rate:.1%}")
    print(f"  Pushes:         {pushes}")
    print(f"  Total profit:   {total_profit:+.1f} units")
    print(f"  ROI:            {roi:+.2f}%")
    print(f"  Avg CLV:        {avg_clv:+.3f} runs")
    print(f"  Break-even WR:  52.4% (at -110)")

    # By season
    print(f"\n  By season:")
    by_season = bets.groupby("season").agg(
        n=("won", "count"),
        wins=("won", "sum"),
        profit=("profit_units", "sum"),
        avg_clv=("clv", "mean"),
    )
    by_season["roi"] = (by_season["profit"] / (by_season["n"] * 1.1)) * 100
    by_season["win_rate"] = by_season["wins"] / by_season["n"]
    print(by_season[["n", "win_rate", "profit", "roi", "avg_clv"]].to_string())

    # By edge bucket
    print(f"\n  By edge size:")
    bets["edge_bucket"] = pd.cut(bets["clv"].abs(),
                                  bins=[0, 0.5, 1.0, 1.5, 2.0, 99],
                                  labels=["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"])
    by_edge = bets.groupby("edge_bucket", observed=True).agg(
        n=("won", "count"),
        win_rate=("won", "mean"),
        profit=("profit_units", "sum"),
    )
    by_edge["roi"] = (by_edge["profit"] / (by_edge["n"] * 1.1)) * 100
    print(by_edge.to_string())


def main():
    parser = argparse.ArgumentParser(description="Backtest MLB totals model")
    parser.add_argument("--model", choices=["ridge", "xgb", "both"], default="both")
    parser.add_argument("--edge", type=float, default=0.75,
                        help="Min abs(predicted - line) to place a bet (default: 0.75)")
    parser.add_argument("--max-edge", type=float, default=1.5,
                        help="Max abs(predicted - line) to place a bet (default: 1.5, use 0 to disable)")
    parser.add_argument("--exclude-seasons", type=int, nargs="+", default=[2020],
                        help="Seasons to exclude from results (default: 2020)")
    args = parser.parse_args()
    max_edge = args.max_edge if args.max_edge > 0 else None

    print("Loading predictions...")
    preds = load_predictions()
    preds_with_lines = preds.dropna(subset=["close_total"])

    if args.exclude_seasons:
        n_before = len(preds_with_lines)
        preds_with_lines = preds_with_lines[~preds_with_lines["season"].isin(args.exclude_seasons)]
        print(f"  {len(preds)} total predictions, {n_before} with closing lines")
        print(f"  Excluding seasons {args.exclude_seasons}: {len(preds_with_lines)} remaining")

    models = []
    if args.model in ("ridge", "both"):
        models.append(("predicted_ridge", "Ridge"))
    if args.model in ("xgb", "both"):
        models.append(("predicted_xgb", "XGBoost"))

    edge_label = f"{args.edge}-{args.max_edge}" if max_edge else f"{args.edge}+"
    for model_col, label in models:
        bets = run_backtest(preds_with_lines, model_col, args.edge, max_edge)
        print_summary(bets, f"{label} | edge {edge_label} runs")

        out_path = PROCESSED_DIR / f"bets_{label.lower()}_{args.edge}.csv"
        bets.to_csv(out_path, index=False)
        print(f"  Bet log saved: {out_path.name}")

    # Ensemble: only bet when Ridge and XGB agree on direction
    if args.model == "both":
        ridge_bets = run_backtest(preds_with_lines, "predicted_ridge", args.edge, max_edge)
        xgb_bets = run_backtest(preds_with_lines, "predicted_xgb", args.edge, max_edge)
        if not ridge_bets.empty and not xgb_bets.empty:
            # Each game_pk appears at most once per model — use ridge as base, join xgb direction
            r = ridge_bets.drop_duplicates("game_pk")
            x = xgb_bets.drop_duplicates("game_pk")[["game_pk", "bet"]].rename(columns={"bet": "xgb_bet"})
            merged = r.merge(x, on="game_pk", how="inner")
            agree = merged[merged["bet"] == merged["xgb_bet"]].drop(columns=["xgb_bet"])
            print_summary(agree, f"Ensemble (Ridge+XGB agree) | edge {edge_label} runs")
            out_path = PROCESSED_DIR / f"bets_ensemble_{args.edge}.csv"
            agree.to_csv(out_path, index=False)
            print(f"  Bet log saved: {out_path.name}")

    # CLV distribution check — did the line move in our favor after open?
    print("\n\nCLV Distribution Check (Ridge):")
    ridge_bets = run_backtest(preds_with_lines, "predicted_ridge", 0.0, max_edge)  # all games within cap
    if not ridge_bets.empty:
        clv_valid = ridge_bets["clv"].dropna()
        if len(clv_valid) > 0:
            print(f"  Games with open+close lines: {len(clv_valid)}")
            print(f"  Mean CLV (line movement):     {clv_valid.mean():+.3f} runs")
            print(f"  % bets with positive CLV:     {(clv_valid > 0).mean():.1%}")
            print(f"  % bets with negative CLV:     {(clv_valid < 0).mean():.1%}")
            print(f"  % bets with flat line (0 CLV): {(clv_valid == 0).mean():.1%}")
            print("  (>50% positive = line moved in your direction after open)")
        else:
            print("  No open line data available for CLV calculation.")


if __name__ == "__main__":
    main()
