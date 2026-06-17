"""
Daily wrapper — scores yesterday's results then generates today's predictions.
Schedule this once at ~11am ET for full daily tracking.

  python src/daily_run.py

Email setup (set these env vars once):
  GMAIL_USER=you@gmail.com
  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
  NOTIFY_TO=you@gmail.com          (optional, defaults to GMAIL_USER)
  ODDS_API_KEY=your_odds_api_key
"""

import sys
import traceback
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import update_results
import predict_today
import notify


def main():
    game_date = date.today().strftime("%Y-%m-%d")

    try:
        # Step 1 — score yesterday's results, get summary
        print("=" * 55)
        print("  STEP 1 — Score yesterday's results")
        print("=" * 55)
        result = update_results.main()
        summary, yesterday_results = result if isinstance(result, tuple) else ({}, [])

        # Step 2 — generate today's predictions, capture flagged bets
        print()
        print("=" * 55)
        print("  STEP 2 — Generate today's predictions")
        print("=" * 55)
        result = predict_today.main()
        today_bets, watch_list = result if isinstance(result, tuple) else (result or [], [])

        # Step 3 — send email
        print()
        print("=" * 55)
        print("  STEP 3 — Send daily summary")
        print("=" * 55)
        notify.notify_daily(summary, today_bets, game_date, yesterday_results, watch_list)

    except Exception as e:
        err = traceback.format_exc()
        print(f"\nERROR during daily run:\n{err}")
        try:
            notify.notify_error(err, game_date)
        except Exception:
            pass


if __name__ == "__main__":
    main()
