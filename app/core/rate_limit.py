from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import parse_rate_limit, settings

_EVICT_EVERY_HITS = 64
_WINDOW_IDLE_SECONDS = 3600


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int


class InMemoryRateLimiter:
    """Process-local sliding window limiter. Swap later if Redis is required."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._windows: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._hits_since_evict = 0

    def hit(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        now = self._clock()
        cutoff = now - window_seconds
        with self._lock:
            self._hits_since_evict += 1
            if self._hits_since_evict >= _EVICT_EVERY_HITS:
                self._evict_locked(now)
            hits = self._windows.get(key)
            if hits is None:
                hits = deque()
                self._windows[key] = hits
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = max(1, int(hits[0] + window_seconds - now) + 1)
                return RateLimitResult(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    retry_after=retry_after,
                )
            hits.append(now)
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=limit - len(hits),
                retry_after=0,
            )

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
            self._hits_since_evict = 0

    def _evict_locked(self, now: float) -> None:
        self._hits_since_evict = 0
        idle_before = now - _WINDOW_IDLE_SECONDS
        stale = [
            key
            for key, hits in self._windows.items()
            if not hits or hits[-1] <= idle_before
        ]
        for key in stale:
            del self._windows[key]


@dataclass
class _FailureState:
    count: int
    locked_until: float | None
    updated_at: float


class LoginProtection:
    """Temporary throttle after repeated failed logins. Never permanently locks."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._failures: dict[str, _FailureState] = {}
        self._lock = threading.Lock()
        self._hits_since_evict = 0

    def allow(self, client_ip: str, email: str) -> bool:
        key = _login_key(client_ip, email)
        now = self._clock()
        with self._lock:
            self._maybe_evict_locked(now)
            state = self._failures.get(key)
            if state is None:
                return True
            if state.locked_until is not None and state.locked_until > now:
                return False
            if state.locked_until is not None or _failure_window_expired(state, now):
                del self._failures[key]
            return True

    def retry_after(self, client_ip: str, email: str) -> int:
        key = _login_key(client_ip, email)
        now = self._clock()
        with self._lock:
            state = self._failures.get(key)
            if state is None or state.locked_until is None:
                return 1
            return max(1, int(state.locked_until - now) + 1)

    def record_failure(self, client_ip: str, email: str) -> None:
        key = _login_key(client_ip, email)
        now = self._clock()
        with self._lock:
            self._maybe_evict_locked(now)
            state = self._failures.get(key)
            if state is None or _should_reset_failures(state, now):
                state = _FailureState(count=0, locked_until=None, updated_at=now)
                self._failures[key] = state
            state.count += 1
            state.updated_at = now
            if state.count >= settings.LOGIN_MAX_FAILED_ATTEMPTS:
                state.locked_until = now + settings.LOGIN_LOCKOUT_SECONDS

    def clear(self, client_ip: str, email: str) -> None:
        key = _login_key(client_ip, email)
        with self._lock:
            self._failures.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()
            self._hits_since_evict = 0

    def _maybe_evict_locked(self, now: float) -> None:
        self._hits_since_evict += 1
        if self._hits_since_evict < _EVICT_EVERY_HITS:
            return
        self._hits_since_evict = 0
        stale = [
            key
            for key, state in self._failures.items()
            if _should_reset_failures(state, now)
        ]
        for key in stale:
            del self._failures[key]


def _failure_window_expired(state: _FailureState, now: float) -> bool:
    return now - state.updated_at >= settings.LOGIN_LOCKOUT_SECONDS


def _should_reset_failures(state: _FailureState, now: float) -> bool:
    if state.locked_until is not None and state.locked_until <= now:
        return True
    return state.locked_until is None and _failure_window_expired(state, now)


def _login_key(client_ip: str, email: str) -> str:
    return f"{client_ip}:{email.strip().lower()}"


rate_limiter = InMemoryRateLimiter()
login_protection = LoginProtection()


def check_endpoint_rate_limit(scope: str, client_ip: str) -> RateLimitResult:
    limit, window_seconds = parse_rate_limit(settings.rate_limit_spec(scope))
    return rate_limiter.hit(f"ep:{scope}:{client_ip}", limit, window_seconds)


def reset_rate_limiters() -> None:
    rate_limiter.reset()
    login_protection.reset()
    rate_limiter._clock = time.monotonic
    login_protection._clock = time.monotonic
