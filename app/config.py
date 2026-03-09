from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "mysql+aiomysql://planner:password@localhost:3306/goal_agent"

    # Kimi Coding (Anthropic-compatible, family/coding plan)
    KIMI_API_KEY: str = "sk-placeholder"
    KIMI_BASE_URL: str = "https://api.kimi.com/coding/"
    KIMI_MODEL_SHORT: str = "k2p5"
    KIMI_MODEL_LONG: str = "k2p5"

    # Telegram
    TELEGRAM_BEST_PAL_BOT_TOKEN: str = ""
    TELEGRAM_GO_GETTER_BOT_TOKEN: str = ""
    TELEGRAM_GROUP_CHAT_ID: str = ""

    # GitHub
    GITHUB_PAT: str = ""
    GITHUB_DATA_REPO: str = "username/study-data-private"
    GITHUB_COMMITTER_NAME: str = "Goal Agent Bot"
    GITHUB_COMMITTER_EMAIL: str = "bot@example.com"

    # App
    APP_SECRET_KEY: str = "change-me-in-production"
    APP_DEBUG: bool = False
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # HMAC request signing (Issue #3)
    # When non-empty, all requests carrying X-Telegram-Chat-Id must include
    # valid X-Request-Timestamp / X-Nonce / X-Signature headers.
    # Leave empty in dev/test environments to skip verification.
    HMAC_SECRET: str = ""

    # Brave Search API
    BRAVE_API_KEY: str = ""

    # Bootstrap admin chat IDs (comma-separated)
    ADMIN_CHAT_IDS: str = ""

    @field_validator("ADMIN_CHAT_IDS", mode="before")
    @classmethod
    def parse_admin_chat_ids(cls, v: str) -> str:
        return v or ""

    def get_admin_chat_ids(self) -> list[int]:
        if not self.ADMIN_CHAT_IDS:
            return []
        return [int(x.strip()) for x in self.ADMIN_CHAT_IDS.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
