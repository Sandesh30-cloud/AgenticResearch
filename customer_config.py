"""Validated tenant policy. Store only non-secret settings in customer JSON files."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class CustomerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    customer_id: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    display_name: str = Field(default="Default customer", validation_alias=AliasChoices("display_name", "company", "name"))
    industry: str | None = None
    customer_type: str | None = None
    team_size: int | None = Field(default=None, ge=1)
    team_description: str | None = Field(default=None, validation_alias=AliasChoices("team_description", "team"))
    business_objective: str | None = None
    current_problem: str | None = None
    expected_output: str | None = None
    model: str = Field(default="gemini-2.5-flash", min_length=1, max_length=120)
    system_prompt: str = "You are an expert research writer. Write clear, structured and insightful reports."
    retry_attempts: int = Field(default=3, ge=1, le=8)
    llm_timeout_seconds: float = Field(default=45, gt=0, le=300)
    search_rate_limit_per_minute: int = Field(default=20, ge=1, le=600)
    max_sources: int = Field(default=5, ge=1, le=20)
    require_human_approval: bool = False

    def research_context(self) -> str:
        """Customer-specific context for retrieval; never includes secrets."""
        values = (
            ("Industry", self.industry),
            ("Customer type", self.customer_type),
            ("Research team size", f"{self.team_size} analysts" if self.team_size else None),
            ("Team", self.team_description),
            ("Business objective", self.business_objective),
            ("Current problem", self.current_problem),
            ("Expected output", self.expected_output),
        )
        return "\n".join(f"{label}: {value}" for label, value in values if value)


@lru_cache(maxsize=1)
def _default_config() -> CustomerConfig:
    return CustomerConfig(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
        search_rate_limit_per_minute=int(os.getenv("SEARCH_RATE_LIMIT_PER_MINUTE", "20")),
    )


def load_customer_config(customer_id: str = "default") -> CustomerConfig:
    """Load and validate {CUSTOMER_CONFIG_DIR}/{customer_id}.json or default policy."""
    if customer_id == "default":
        return _default_config()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", customer_id):
        raise ValueError(
            "Customer profile must be a short ID such as 'acme_research'. "
            "Put company details in customer_configs/<id>.json."
        )
    config_dir = Path(os.getenv("CUSTOMER_CONFIG_DIR", "customer_configs")).resolve()
    candidate = (config_dir / f"{customer_id}.json").resolve()
    if candidate.parent != config_dir or not candidate.is_file():
        raise ValueError(f"No configuration found for customer '{customer_id}'")
    with candidate.open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
    config = CustomerConfig.model_validate(raw_config)
    if config.customer_id != customer_id:
        raise ValueError("Customer configuration ID does not match the requested customer")
    return config


def resolve_customer_profile(profile_input: str | None) -> tuple[CustomerConfig, str]:
    """Resolve a saved profile ID or accept a free-form customer context brief."""
    value = (profile_input or "").strip()
    if not value:
        config = load_customer_config("default")
        return config, config.research_context()
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value):
        config = load_customer_config(value)
        return config, config.research_context()
    config = load_customer_config("default")
    context = "\n".join(part for part in (config.research_context(), value) if part)
    return config, context
