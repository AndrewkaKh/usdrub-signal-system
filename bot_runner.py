"""Telegram bot entry point.

Usage:
    python bot_runner.py --morning                   # send morning report now
    python bot_runner.py --evening                   # send evening report for latest date
    python bot_runner.py --evening --date 2026-04-09 # send for a specific date
    python bot_runner.py --schedule                  # run on a daily schedule (MVP)

"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Load .env before anything else so bot.config can read the variables
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger('bot_runner')


# ---------------------------------------------------------------------------
# Report senders
# ---------------------------------------------------------------------------

def send_morning_report(date_str: str | None = None) -> None:
    from bot.client import send_message
    from bot.pipeline import morning_payload
    from bot.reports import format_morning

    logger.info('Building morning report...')
    try:
        payload = morning_payload(date_str)
        text = format_morning(payload)
        ok = send_message(text)
        if ok:
            logger.info('Morning report sent.')
        else:
            logger.error('Failed to send morning report.')
    except Exception as exc:
        logger.exception('Unexpected error in morning report: %s', exc)
        _send_error_alert(f'Утренний отчёт: {exc}')


def send_evening_report(date_str: str | None = None) -> None:
    from bot.client import send_message, send_photo
    from bot.pipeline import evening_payload
    from bot.reports import format_evening
    from bot.state import is_evening_sent, mark_evening_sent
    from datetime import date

    target_date = date_str or date.today().isoformat()

    if is_evening_sent(target_date):
        logger.info('Evening report for %s already sent, skipping.', target_date)
        return

    logger.info('Building evening report for %s...', target_date)
    try:
        payload, png_path = evening_payload(target_date)
        text = format_evening(payload)

        if png_path and Path(png_path).exists():
            # Send photo with report text as caption (combined post)
            ok = send_photo(png_path, caption=text)
            if ok:
                logger.info('Evening report with chart sent.')
            else:
                logger.error('Failed to send evening report with chart.')
                return
        else:
            # No chart available — fall back to text-only
            ok = send_message(text)
            if ok:
                logger.info('Evening report sent (no chart).')
            else:
                logger.error('Failed to send evening report.')
                return

        mark_evening_sent(target_date)

    except Exception as exc:
        logger.exception('Unexpected error in evening report: %s', exc)
        _send_error_alert(f'Вечерний отчёт: {exc}')


def _send_error_alert(msg: str) -> None:
    """Best-effort error notification to the channel."""
    try:
        from bot.client import send_message
        send_message(f'⚠️ *Ошибка бота*: `{msg}`')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='USD/RUB IV Report Bot')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--morning', action='store_true', help='Send morning report now')
    group.add_argument('--evening', action='store_true', help='Send evening report now')
    group.add_argument(
        '--schedule',
        action='store_true',
        help='Run with daily schedule (MVP — prefer Task Scheduler for production)',
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date for --evening in YYYY-MM-DD format (default: today)',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.morning:
        send_morning_report()

    elif args.evening:
        send_evening_report(args.date)

    elif args.schedule:
        import schedule
        import time

        logger.info(
            'Scheduler started. Morning: 10:00, Evening: 20:30 (local time).\n'
            'NOTE: This is an MVP runner. For production, use Task Scheduler or systemd.'
        )

        schedule.every().day.at('10:00').do(send_morning_report)
        schedule.every().day.at('20:30').do(send_evening_report)

        while True:
            schedule.run_pending()
            time.sleep(30)


if __name__ == '__main__':
    main()
