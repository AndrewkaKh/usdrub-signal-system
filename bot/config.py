"""Bot configuration via Pydantic Settings.

Reads from environment variables or a .env file automatically.
Uses lazy initialization — settings are validated on first call to get_settings(),
not at import time, so modules can be imported without a .env for testing.

Usage:
    from bot.config import get_settings
    s = get_settings()
    token = s.telegram_bot_token
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

    telegram_bot_token: str = Field(
        ...,
        description='Telegram Bot API token from @BotFather',
    )
    telegram_channel_id: str = Field(
        ...,
        description='Channel ID (e.g. -1001234567890) or @username',
    )

    # File paths (have defaults, no .env required)
    dataset_csv: str = 'data/exports/model_dataset_daily.csv'
    db_path: str = 'data/backfill/moex_backfill.sqlite3'
    smile_csv: str = 'data/exports/iv_smile_daily.csv'
    smile_metrics_csv: str = 'data/exports/iv_smile_metrics.csv'
    smile_tmp_dir: str = 'data/exports/tmp'
    bot_state_path: str = 'data/exports/bot_state.json'


@lru_cache(maxsize=1)
def get_settings() -> BotSettings:
    """Return the validated settings singleton.

    Raises pydantic_core.ValidationError if required env vars are missing.
    Call this only from functions that actually need the config (not at import time).
    """
    return BotSettings()
