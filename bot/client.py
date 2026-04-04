"""Telegram Bot API client via plain requests (no heavy framework).

Only sends — no polling/webhooks needed for a reporting bot.
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

from bot.config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT = (5, 30)  # (connect, read) seconds


def _base() -> str:
    return f'https://api.telegram.org/bot{get_settings().telegram_bot_token}'


def send_message(text: str, parse_mode: str = 'Markdown') -> bool:
    """Send a text message to the configured channel.

    Returns True on success, False on failure (error is logged).
    """
    try:
        resp = requests.post(
            f'{_base()}/sendMessage',
            json={'chat_id': get_settings().telegram_channel_id, 'text': text, 'parse_mode': parse_mode},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error('send_message failed: %s', exc)
        return False


def send_photo(photo_path: str | Path, caption: str = '', parse_mode: str = 'Markdown') -> bool:
    """Send a photo file to the configured channel.

    Returns True on success, False on failure (error is logged).
    """
    path = Path(photo_path)
    if not path.exists():
        logger.error('send_photo: file not found: %s', path)
        return False
    try:
        with path.open('rb') as fh:
            resp = requests.post(
                f'{_base()}/sendPhoto',
                data={'chat_id': get_settings().telegram_channel_id, 'caption': caption, 'parse_mode': parse_mode},
                files={'photo': (path.name, fh, 'image/png')},
                timeout=_TIMEOUT,
            )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error('send_photo failed: %s', exc)
        return False
