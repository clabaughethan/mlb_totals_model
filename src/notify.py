"""
Send daily summary email via Gmail SMTP.

Setup (one-time):
  1. Go to myaccount.google.com -> Security -> 2-Step Verification -> App passwords
  2. Generate an app password for "Mail"
  3. Set environment variables:
       GMAIL_USER=you@gmail.com
       GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

  On PythonAnywhere, add to ~/.bashrc:
       export GMAIL_USER=you@gmail.com
       export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
       export NOTIFY_TO=you@gmail.com   # defaults to GMAIL_USER if not set
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date


def _fmt_block(stats: dict, unit_sizes: list) -> str:
    if not stats:
        return "  No scored bets.\n"
    w, l, p = stats["wins"], stats["losses"], stats["pushes"]
    lines = [
        f"  Record:   {w}-{l}-{p}  ({stats['win_rate']:.1%} win rate)",
        f"  Units:    {stats['profit_units']:+.2f}",
        f"  ROI:      {stats['roi']:+.1f}%",
    ]
    for unit in unit_sizes:
        dollar = stats["profit_units"] * unit
        lines.append(f"  ${unit:>3}/bet: ${dollar:+.2f}")
    return "\n".join(lines)


def build_email_body(summary: dict, today_bets: list, game_date: str,
                     yesterday_results: list = None, watch_list: list = None,
                     unit_sizes: list = None) -> str:
    if unit_sizes is None:
        unit_sizes = [10, 20, 25, 50, 100]

    lines = [
        f"MLB Totals Model — {game_date}",
        "=" * 50,
        "",
    ]

    # Today's bets
    if today_bets:
        lines.append(f"TODAY'S BETS ({len(today_bets)} flagged)")
        lines.append("─" * 50)
        for b in today_bets:
            sp_line = f"  {b.get('away_sp','TBD')} vs {b.get('home_sp','TBD')}"
            wx = b.get("weather", "")
            lines.append(
                f"  {b['bet']} {b['away']} @ {b['home']}"
                f"  |  Pred: {b['predicted']:.1f}  Line: {b['line']}  Edge: {b['edge']:+.2f}"
            )
            lines.append(sp_line)
            if wx:
                lines.append(f"  {wx}")
        lines.append("")
    else:
        lines.append("No bets flagged today.")
        lines.append("")

    # Watch list
    if watch_list:
        lines.append(f"WATCH LIST — edge >{1.5} (not betting)")
        lines.append("─" * 50)
        for b in watch_list:
            lines.append(
                f"  {b['bet']} {b['away']} @ {b['home']}"
                f"  |  Pred: {b['predicted']:.1f}  Line: {b['line']}  Edge: {b['edge']:+.2f}"
            )
            lines.append(f"  {b.get('away_sp','TBD')} vs {b.get('home_sp','TBD')}")
        lines.append("")

    # Yesterday's results
    lines.append("YESTERDAY'S RESULTS")
    lines.append("─" * 50)
    if yesterday_results:
        for r in yesterday_results:
            outcome = "WIN" if r["won"] else ("PUSH" if r["push"] else "LOSS")
            profit = r["profit_units"]
            lines.append(
                f"  {outcome}  {r['bet']} {r['away']} @ {r['home']}"
                f"  |  Line: {r['line']}  Actual: {r['actual']}  ({profit:+.1f}u)"
            )
    else:
        lines.append("  No bets yesterday.")
    lines.append("")

    # Performance summary
    if summary:
        for label, key in [("OVERALL", "overall"),
                            ("LAST 30 DAYS", "l30"),
                            ("LAST 7 DAYS", "l7")]:
            lines.append(label)
            lines.append("─" * 50)
            lines.append(_fmt_block(summary.get(key), unit_sizes))
            lines.append("")

        # By direction
        lines.append("BY DIRECTION")
        lines.append("─" * 50)
        for direction in ["OVER", "UNDER"]:
            stats = summary.get("by_direction", {}).get(direction)
            if stats:
                lines.append(f"  {direction}:")
                lines.append(_fmt_block(stats, unit_sizes))
        lines.append("")

        # By edge bucket
        lines.append("BY EDGE BUCKET")
        lines.append("─" * 50)
        for bucket, stats in summary.get("by_edge", {}).items():
            if stats:
                lines.append(f"  Edge {bucket}:")
                lines.append(_fmt_block(stats, unit_sizes))
        lines.append("")

        # Recent bets table
        recent = summary.get("recent")
        if recent is not None and not recent.empty:
            lines.append("RECENT BETS (last 10)")
            lines.append("─" * 50)
            lines.append(recent.to_string(index=False))
            lines.append("")
    else:
        lines.append("No completed bets yet — tracking starts after first results.")

    lines.append("─" * 50)
    lines.append("MLB Totals Model  |  automated daily run")

    return "\n".join(lines)


def send_email(subject: str, body: str) -> bool:
    """Send plain-text email via Gmail SMTP. Returns True on success."""
    gmail_user = os.environ.get("GMAIL_USER", "")
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    to_addr = os.environ.get("NOTIFY_TO", gmail_user)

    if not gmail_user or not app_password:
        print("  Email skipped — GMAIL_USER / GMAIL_APP_PASSWORD not set.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to_addr
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, app_password)
            server.sendmail(gmail_user, to_addr, msg.as_string())
        print(f"  Email sent to {to_addr}")
        return True
    except Exception as e:
        print(f"  Email failed: {e}")
        return False


def notify_daily(summary: dict, today_bets: list, game_date: str,
                 yesterday_results: list = None, watch_list: list = None):
    """Build and send the daily summary email."""
    subject = f"MLB Totals Model — {game_date}"
    body = build_email_body(summary, today_bets, game_date, yesterday_results, watch_list)

    print(f"\n  Sending daily summary email...")
    send_email(subject, body)


def notify_error(error_msg: str, game_date: str):
    """Send a short error alert if the daily run fails."""
    subject = f"MLB Model ERROR — {game_date}"
    body = f"Daily run failed on {game_date}:\n\n{error_msg}"
    send_email(subject, body)
