"""Validated contracts shared by the pipeline and future REST API."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=3, max_length=500)
    customer_id: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    customer_context: str = Field(default="", max_length=4000)
    research_questions: list[str] = Field(default_factory=list, max_length=20)
    deliverables: list[str] = Field(default_factory=list, max_length=20)


class SourceRecord(BaseModel):
    url: HttpUrl
    title: str | None = None
    publication_date: date | None = None
    accessed: bool = False


class ExecutionRecord(BaseModel):
    agent: str
    status: Literal["pending", "running", "completed", "failed"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    attempts: int = Field(default=0, ge=0)
    error: dict | None = None


class ResearchResponse(BaseModel):
    topic: str
    customer_id: str = "default"
    customer_context: str = ""
    research_questions: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    search_results: str = ""
    scraped_content: str = ""
    report: str = ""
    feedback: str = ""
    executions: dict[str, ExecutionRecord] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
