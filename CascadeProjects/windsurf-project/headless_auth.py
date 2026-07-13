#!/usr/bin/env python3
"""Short-lived, in-memory browser sessions for Weaver headless v2."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass


SESSION_COOKIE_NAME = "__Host-weaver-v2"
DEFAULT_SESSION_TTL_SECONDS = 15 * 60
MAX_SESSION_LIFETIME_SECONDS = 8 * 60 * 60


@dataclass(frozen=True)
class SessionGrant:
    token: str
    csrf_token: str
    expires_at: float


@dataclass(frozen=True)
class SessionRenewal:
    csrf_token: str
    expires_at: float


@dataclass
class _SessionRecord:
    csrf_digest: str
    created_at: float
    expires_at: float
    absolute_expires_at: float
    last_seen_at: float


class HeadlessSessionStore:
    """Bounded process-local session registry that stores only token digests."""

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        max_lifetime_seconds: int = MAX_SESSION_LIFETIME_SECONDS,
        max_sessions: int = 128,
        clock=time.time,
    ) -> None:
        self.ttl_seconds = min(max(int(ttl_seconds), 60), 3_600)
        self.max_lifetime_seconds = min(
            max(int(max_lifetime_seconds), self.ttl_seconds),
            MAX_SESSION_LIFETIME_SECONDS,
        )
        self.max_sessions = min(max(int(max_sessions), 8), 2_048)
        self._clock = clock
        self._lock = asyncio.Lock()
        self._sessions: dict[str, _SessionRecord] = {}

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _prune(self, now: float) -> None:
        self._sessions = {
            digest: record
            for digest, record in self._sessions.items()
            if record.expires_at >= now and record.absolute_expires_at >= now
        }

    async def issue(self) -> SessionGrant:
        now = float(self._clock())
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(24)
        token_digest = self._digest(token)
        record = _SessionRecord(
            csrf_digest=self._digest(csrf_token),
            created_at=now,
            expires_at=now + self.ttl_seconds,
            absolute_expires_at=now + self.max_lifetime_seconds,
            last_seen_at=now,
        )
        async with self._lock:
            self._prune(now)
            if len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].last_seen_at)
                self._sessions.pop(oldest, None)
            self._sessions[token_digest] = record
        return SessionGrant(token=token, csrf_token=csrf_token, expires_at=record.expires_at)

    async def authenticate(
        self,
        token: str,
        *,
        csrf_token: str = "",
        require_csrf: bool = False,
    ) -> bool:
        if not token:
            return False
        now = float(self._clock())
        token_digest = self._digest(token)
        async with self._lock:
            self._prune(now)
            record = self._sessions.get(token_digest)
            if record is None:
                return False
            if require_csrf:
                supplied = self._digest(csrf_token) if csrf_token else ""
                if not hmac.compare_digest(supplied, record.csrf_digest):
                    return False
            record.last_seen_at = now
            return True

    async def renew(self, token: str, csrf_token: str) -> SessionRenewal | None:
        if not token or not csrf_token:
            return None
        now = float(self._clock())
        token_digest = self._digest(token)
        supplied_csrf = self._digest(csrf_token)
        async with self._lock:
            self._prune(now)
            record = self._sessions.get(token_digest)
            if record is None or not hmac.compare_digest(supplied_csrf, record.csrf_digest):
                return None
            next_csrf = secrets.token_urlsafe(24)
            record.csrf_digest = self._digest(next_csrf)
            record.expires_at = min(now + self.ttl_seconds, record.absolute_expires_at)
            record.last_seen_at = now
            return SessionRenewal(csrf_token=next_csrf, expires_at=record.expires_at)

    async def revoke(self, token: str, csrf_token: str) -> bool:
        if not token or not csrf_token:
            return False
        token_digest = self._digest(token)
        supplied_csrf = self._digest(csrf_token)
        async with self._lock:
            record = self._sessions.get(token_digest)
            if record is None or not hmac.compare_digest(supplied_csrf, record.csrf_digest):
                return False
            self._sessions.pop(token_digest, None)
            return True

    async def snapshot(self) -> dict[str, int]:
        now = float(self._clock())
        async with self._lock:
            self._prune(now)
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds,
                "max_lifetime_seconds": self.max_lifetime_seconds,
            }
