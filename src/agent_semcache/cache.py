"""SQLite-backed lexical prompt cache with TTL, a max-entries limit (least recently used goes first) and real hit / miss counters."""
from __future__ import annotations

import functools
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .similarity import DEFAULT_MEASURE, MEASURES, normalize, similarity_norm

DEFAULT_THRESHOLD = 0.85
DEFAULT_MAX_ENTRIES = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ns TEXT NOT NULL, ctx TEXT NOT NULL, norm TEXT NOT NULL,
  prompt TEXT NOT NULL, response TEXT NOT NULL,
  created REAL NOT NULL, last_used REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0,
  UNIQUE (ns, ctx, norm)
);
CREATE TABLE IF NOT EXISTS counters (
  ns TEXT NOT NULL, name TEXT NOT NULL, value INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (ns, name)
);
"""


@dataclass(frozen=True)
class Hit:
    score: float          # similarity between the new prompt and the stored one, 0..1
    prompt: str           # the stored prompt that matched
    response: str
    age_s: float          # seconds since that entry was stored
    hits: int             # times this entry has been served (including this lookup, if it was a counted lookup)


def default_db_path() -> str:
    return os.environ.get("AGENT_SEMCACHE_DB") or str(Path.home() / ".cache" / "agent-semcache" / "cache.sqlite3")


class SemCache:
    """One cache file, one namespace. Entries are matched only inside the same namespace and the same context string."""

    def __init__(self, path: str = ":memory:", *, threshold: float = DEFAULT_THRESHOLD, measure: str = DEFAULT_MEASURE,
                 ttl: Optional[float] = None, max_entries: Optional[int] = DEFAULT_MAX_ENTRIES, namespace: str = "default",
                 clock: Callable[[], float] = time.time):
        _check_threshold(threshold)
        if measure not in MEASURES:
            raise ValueError(f"unknown measure {measure!r}; choose one of {sorted(MEASURES)}")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be a positive number of seconds or None")
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1 or None")
        self.path, self.threshold, self.measure, self.ttl = path, threshold, measure, ttl
        self.max_entries, self.namespace, self._clock = max_entries, namespace, clock
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            path = str(Path(path).expanduser())
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # -- internals ---------------------------------------------------------------------------------------------
    def _bump(self, name: str, by: int = 1) -> None:
        self._db.execute("INSERT INTO counters (ns, name, value) VALUES (?, ?, ?) "
                         "ON CONFLICT(ns, name) DO UPDATE SET value = value + excluded.value", (self.namespace, name, by))

    def _purge_expired(self, now: float) -> int:
        if self.ttl is None:
            return 0
        n = self._db.execute("DELETE FROM entries WHERE ns = ? AND created <= ?", (self.namespace, now - self.ttl)).rowcount
        if n:
            self._bump("expired", n)
        return n

    def _best(self, norm: str, ctx: str, now: float):
        best = None
        for row in self._db.execute("SELECT id, norm, prompt, response, created, hits FROM entries WHERE ns = ? AND ctx = ?",
                                    (self.namespace, ctx)):
            score = similarity_norm(norm, row[1], self.measure)
            # highest score wins; on a tie the most recently stored entry wins
            if best is None or score > best[0] or (score == best[0] and row[4] > best[1][4]):
                best = (score, row)
        return best

    # -- public API --------------------------------------------------------------------------------------------
    def lookup(self, prompt: str, *, threshold: Optional[float] = None, context: str = "") -> Optional[Hit]:
        """Return the stored answer whose prompt scores >= threshold against `prompt`, else None. Counts one hit or one miss."""
        th = self.threshold if threshold is None else threshold
        _check_threshold(th)
        norm = normalize(prompt)
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            best = self._best(norm, context, now)
            if best is not None and best[0] >= th:
                score, row = best
                self._db.execute("UPDATE entries SET last_used = ?, hits = hits + 1 WHERE id = ?", (now, row[0]))
                self._bump("hits")
                self._db.commit()
                return Hit(score=score, prompt=row[2], response=row[3], age_s=max(0.0, now - row[4]), hits=row[5] + 1)
            self._bump("misses")
            self._db.commit()
            return None

    def nearest(self, prompt: str, *, context: str = "") -> Optional[Hit]:
        """The closest stored prompt whatever its score (for tuning a threshold). Changes no counter and no entry."""
        norm = normalize(prompt)
        with self._lock:
            now = self._clock()
            live_cutoff = None if self.ttl is None else now - self.ttl
            best = None
            for row in self._db.execute("SELECT id, norm, prompt, response, created, hits FROM entries WHERE ns = ? AND ctx = ?",
                                        (self.namespace, context)):
                if live_cutoff is not None and row[4] <= live_cutoff:
                    continue
                score = similarity_norm(norm, row[1], self.measure)
                if best is None or score > best[0] or (score == best[0] and row[4] > best[1][4]):
                    best = (score, row)
            if best is None:
                return None
            score, row = best
            return Hit(score=score, prompt=row[2], response=row[3], age_s=max(0.0, now - row[4]), hits=row[5])

    def store(self, prompt: str, response: str, *, context: str = "") -> None:
        """Save an answer. The same normalized prompt (in the same namespace and context) is overwritten and its age restarts."""
        if not isinstance(response, str):
            raise TypeError(f"response must be str, got {type(response).__name__}")
        norm = normalize(prompt)
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            self._db.execute(
                "INSERT INTO entries (ns, ctx, norm, prompt, response, created, last_used, hits) VALUES (?, ?, ?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(ns, ctx, norm) DO UPDATE SET prompt = excluded.prompt, response = excluded.response, "
                "created = excluded.created, last_used = excluded.last_used, hits = 0",
                (self.namespace, context, norm, prompt, response, now, now))
            self._bump("stores")
            if self.max_entries is not None:
                over = self._db.execute("SELECT COUNT(*) FROM entries WHERE ns = ?", (self.namespace,)).fetchone()[0] - self.max_entries
                if over > 0:
                    self._db.execute("DELETE FROM entries WHERE id IN (SELECT id FROM entries WHERE ns = ? "
                                     "ORDER BY last_used ASC, id ASC LIMIT ?)", (self.namespace, over))
                    self._bump("evictions", over)
            self._db.commit()

    def stats(self) -> dict:
        """Real counters since the cache file was created (or last `clear(reset_stats=True)`), for this namespace."""
        with self._lock:
            self._purge_expired(self._clock())
            self._db.commit()
            c = dict(self._db.execute("SELECT name, value FROM counters WHERE ns = ?", (self.namespace,)).fetchall())
            entries = self._db.execute("SELECT COUNT(*) FROM entries WHERE ns = ?", (self.namespace,)).fetchone()[0]
        hits, misses = c.get("hits", 0), c.get("misses", 0)
        return {"namespace": self.namespace, "hits": hits, "misses": misses, "lookups": hits + misses,
                "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None,
                "entries": entries, "max_entries": self.max_entries, "ttl_s": self.ttl,
                "stores": c.get("stores", 0), "evictions": c.get("evictions", 0), "expired": c.get("expired", 0),
                "threshold": self.threshold, "measure": self.measure}

    def clear(self, *, reset_stats: bool = False) -> int:
        """Delete every entry of this namespace (nothing else). Returns how many were removed."""
        with self._lock:
            n = self._db.execute("DELETE FROM entries WHERE ns = ?", (self.namespace,)).rowcount
            if reset_stats:
                self._db.execute("DELETE FROM counters WHERE ns = ?", (self.namespace,))
            self._db.commit()
            return n

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _check_threshold(th) -> None:
    if isinstance(th, bool) or not isinstance(th, (int, float)) or not 0.0 <= th <= 1.0:
        raise ValueError(f"threshold must be a number between 0 and 1, got {th!r}")


def _context(args: tuple, kwargs: dict) -> str:
    """Extra call arguments must match exactly for a cached answer to be reused."""
    if not args and not kwargs:
        return ""
    return json.dumps([list(args), sorted(kwargs.items())], default=repr, sort_keys=True)


def cached(threshold=DEFAULT_THRESHOLD, *, path: Optional[str] = None, ttl: Optional[float] = None,
           max_entries: Optional[int] = DEFAULT_MAX_ENTRIES, measure: str = DEFAULT_MEASURE,
           namespace: Optional[str] = None, clock: Callable[[], float] = time.time):
    """Wrap `fn(prompt, ...) -> str`. The first positional argument (or `prompt=`) is the cache key; any other
    arguments must be identical for a stored answer to be reused. Usable as `@cached` or `@cached(threshold=0.9)`.
    The wrapper has `.cache` (the SemCache) and `.cache_info()` (its stats)."""
    def decorate(fn):
        cache = SemCache(path or default_db_path(), threshold=th, measure=measure, ttl=ttl, max_entries=max_entries,
                         namespace=namespace or f"{fn.__module__}.{fn.__qualname__}", clock=clock)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if args:
                prompt, rest, kw = args[0], args[1:], kwargs
            elif "prompt" in kwargs:
                prompt, rest, kw = kwargs["prompt"], (), {k: v for k, v in kwargs.items() if k != "prompt"}
            else:
                raise TypeError("a cached function needs its prompt as the first positional argument or prompt=")
            if not isinstance(prompt, str):
                raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
            ctx = _context(rest, kw)
            hit = cache.lookup(prompt, context=ctx)
            if hit is not None:
                return hit.response
            out = fn(*args, **kwargs)
            if not isinstance(out, str):
                raise TypeError(f"a cached function must return str, got {type(out).__name__}")
            cache.store(prompt, out, context=ctx)
            return out

        wrapper.cache = cache
        wrapper.cache_info = cache.stats
        return wrapper

    if callable(threshold):          # bare @cached
        fn, th = threshold, DEFAULT_THRESHOLD
        return decorate(fn)
    th = threshold
    _check_threshold(th)
    return decorate
