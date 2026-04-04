"""Idempotency state for the evening report.

Stores which dates have already had a report sent so that re-runs
(e.g. after a crash or manual test) don't produce duplicate messages.

State file: data/exports/bot_state.json
Format:     {"evening_sent": ["2026-04-09", "2026-04-10", ...]}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from bot.config import get_settings

logger = logging.getLogger(__name__)


def _state_path() -> Path:
    return Path(get_settings().bot_state_path)


def _load() -> dict:
    if _state_path().exists():
        try:
            return json.loads(_state_path().read_text(encoding='utf-8'))
        except Exception as exc:
            logger.warning('Could not read bot_state.json: %s', exc)
    return {'evening_sent': []}


def _save(state: dict) -> None:
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=2), encoding='utf-8')


def is_evening_sent(date_str: str) -> bool:
    """Return True if an evening report for date_str was already sent."""
    return date_str in _load().get('evening_sent', [])


def mark_evening_sent(date_str: str) -> None:
    """Record that the evening report for date_str has been sent."""
    state = _load()
    sent = state.setdefault('evening_sent', [])
    if date_str not in sent:
        sent.append(date_str)
        # Keep only last 30 dates to prevent unbounded growth
        state['evening_sent'] = sent[-30:]
        _save(state)
