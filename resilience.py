"""Shared retry, rate limiting, and execution tracking for pipeline stages."""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class StageError:
    code: str
    message: str
    retryable: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentExecution:
    agent: str
    status: str = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    duration_seconds: float | None = None
    attempts: int = 0
    error: dict[str, Any] | None = None


class RateLimiter:
    """Thread-safe minimum-interval limiter (per process)."""
    def __init__(self, calls_per_minute: float):
        self.interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0
        self._lock = threading.Lock()
        self._next_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_call - now)
            self._next_call = max(now, self._next_call) + self.interval
        if delay:
            time.sleep(delay)


_SEARCH_LIMITERS: dict[float, RateLimiter] = {}
_SEARCH_LIMITER_LOCK = threading.Lock()


def rate_limit_search(calls_per_minute: float | None = None) -> None:
    limit = calls_per_minute or float(os.getenv("SEARCH_RATE_LIMIT_PER_MINUTE", "20"))
    with _SEARCH_LIMITER_LOCK:
        limiter = _SEARCH_LIMITERS.setdefault(limit, RateLimiter(limit))
    limiter.wait()


def error_from_exception(exc: Exception, *, timeout: bool = False) -> StageError:
    if timeout or isinstance(exc, TimeoutError):
        return StageError("timeout", str(exc) or "Operation timed out", True)
    name = type(exc).__name__.lower()
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    retryable = (status == 429 or (isinstance(status, int) and status >= 500) or
                 any(token in name for token in ("timeout", "connection", "ratelimit", "temporar", "serviceunavailable")))
    code = "rate_limited" if "rate" in name else "upstream_error"
    return StageError(code, str(exc) or type(exc).__name__, retryable,
                      {"http_status": status} if status is not None else {})


def run_stage(name: str, operation: Callable[[], T], *, executions: dict[str, dict] | None = None,
              max_attempts: int = 3, base_delay: float = 0.5, max_delay: float = 8.0) -> T:
    """Run a stage with bounded exponential backoff and record each attempt."""
    record = AgentExecution(agent=name, status="running", started_at=time.time())
    if executions is not None:
        executions[name] = asdict(record)
    for attempt in range(1, max_attempts + 1):
        record.attempts = attempt
        try:
            value = operation()
            record.status = "completed"
            record.finished_at = time.time()
            record.duration_seconds = round(record.finished_at - record.started_at, 3)
            if executions is not None:
                executions[name] = asdict(record)
            return value
        except Exception as exc:
            error = error_from_exception(exc)
            record.error = error.as_dict()
            if not error.retryable or attempt == max_attempts:
                record.status = "failed"
                record.finished_at = time.time()
                record.duration_seconds = round(record.finished_at - record.started_at, 3)
                if executions is not None:
                    executions[name] = asdict(record)
                raise
            time.sleep(min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.15))
    raise RuntimeError("unreachable")


def failed_execution(name: str, exc: Exception) -> dict[str, Any]:
    now = time.time()
    return asdict(AgentExecution(agent=name, status="failed", finished_at=now,
                                 error=error_from_exception(exc).as_dict()))
