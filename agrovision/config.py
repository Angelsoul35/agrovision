"""Runtime configuration, loaded from environment variables with the ``AV_`` prefix."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AV_", env_file=".env", extra="ignore")

    # Confidence below which a prediction is routed to manual (agronomist) review
    # instead of being reported as a firm verdict.
    classifier_min_confidence: float = 0.55

    # Deterministic "clearly healthy" rule. A leaf whose green fraction is at or
    # above this threshold *and* whose diseased-spot coverage is at or below
    # ``rule_max_spot_coverage`` is called healthy without consulting the model.
    rule_green_fraction: float = 0.47
    rule_max_spot_coverage: float = 0.02


def get_settings() -> Settings:
    return Settings()
