#!/usr/bin/env python3
"""
akashic_hub.py — Weaver's Akashic Entanglement Layer
=====================================================
A shared in-process vector state space that replaces linear inter-lobe
messaging with zero-latency dimensional reads.

Every lobe writes its current state as a fixed-size numpy vector into the
hub.  Any other lobe can instantly read the full dimensional state — or
query by cosine similarity — without serialization, network hops, or
pub/sub round-trips.

The hub also maintains a temporal trace (last N snapshots per lobe) so the
Liquid Fracture Engine can observe how lobe states evolve over time.

Design targets:
    • Write latency:   < 0.5 ms   (numpy array copy)
    • Read latency:    < 0.1 ms   (numpy view / reference)
    • Similarity query: < 1.0 ms   (vectorized cosine on 256-d × 16 lobes)

Usage:
    hub = AkashicHub(dim=256, trace_depth=32)
    await hub.write("quantum_soul", state_vector)
    snapshot = hub.read("quantum_soul")          # instant numpy view
    full     = hub.read_all()                    # {lobe_id: vector}
    similar  = hub.query(probe_vector, top_k=3)  # cosine-ranked lobes
    trace    = hub.temporal_trace("quantum_soul") # last N states
"""

import asyncio
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Akashic Hub ───────────────────────────────────────────────────────────────

class AkashicHub:
    """Zero-latency shared vector state for all Weaver lobes."""

    def __init__(self, dim: int = 256, trace_depth: int = 32):
        self.dim = dim
        self.trace_depth = trace_depth

        # Current state vectors — { lobe_id: np.ndarray(dim,) }
        self._state: Dict[str, np.ndarray] = {}

        # Temporal traces — { lobe_id: deque of (timestamp, vector) }
        self._traces: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.trace_depth)
        )

        # Metadata — { lobe_id: dict }  (arbitrary per-lobe metadata)
        self._meta: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Write timestamps — { lobe_id: float }
        self._timestamps: Dict[str, float] = {}

        # Async lock for write-consistency (reads are lock-free)
        self._lock = asyncio.Lock()

        # Event listeners — notified on every write
        self._listeners: List[asyncio.Queue] = []

    # ── Write ─────────────────────────────────────────────────────────────

    async def write(self, lobe_id: str, vector: np.ndarray,
                    meta: Optional[Dict[str, Any]] = None) -> float:
        """Write a lobe's current dimensional state into the hub.

        Args:
            lobe_id: Unique lobe identifier (e.g. "quantum_soul").
            vector:  State vector.  Must be shape (dim,) or will be
                     zero-padded / truncated to fit.
            meta:    Optional metadata dict attached to this write.

        Returns:
            Write latency in milliseconds.
        """
        t0 = time.perf_counter_ns()

        v = self._coerce(vector)
        ts = time.time()

        async with self._lock:
            self._state[lobe_id] = v
            self._traces[lobe_id].append((ts, v.copy()))
            self._timestamps[lobe_id] = ts
            if meta:
                self._meta[lobe_id].update(meta)

        # Notify listeners
        event = {"lobe_id": lobe_id, "ts": ts, "vector": v}
        for q in self._listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # drop if listener is slow

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
        return latency_ms

    # ── Read (lock-free) ──────────────────────────────────────────────────

    def read(self, lobe_id: str) -> Optional[np.ndarray]:
        """Read a lobe's current state.  Returns None if lobe not yet written."""
        return self._state.get(lobe_id)

    def read_all(self) -> Dict[str, np.ndarray]:
        """Read every lobe's current state.  Returns a shallow copy of the map."""
        return dict(self._state)

    def read_meta(self, lobe_id: str) -> Dict[str, Any]:
        """Read metadata for a lobe."""
        return dict(self._meta.get(lobe_id, {}))

    def active_lobes(self) -> List[str]:
        """Return list of lobe IDs that have written at least once."""
        return list(self._state.keys())

    def age(self, lobe_id: str) -> Optional[float]:
        """Seconds since the lobe's last write, or None if never written."""
        ts = self._timestamps.get(lobe_id)
        if ts is None:
            return None
        return time.time() - ts

    # ── Temporal Trace ────────────────────────────────────────────────────

    def temporal_trace(self, lobe_id: str) -> List[Tuple[float, np.ndarray]]:
        """Return the last N (timestamp, vector) snapshots for a lobe."""
        return list(self._traces.get(lobe_id, []))

    def temporal_matrix(self, lobe_id: str) -> Optional[np.ndarray]:
        """Return trace as a (N, dim) matrix for ODE / liquid processing."""
        trace = self._traces.get(lobe_id)
        if not trace:
            return None
        return np.stack([v for _, v in trace])

    def temporal_deltas(self, lobe_id: str) -> Optional[np.ndarray]:
        """Return the first-order differences of the trace (velocity)."""
        mat = self.temporal_matrix(lobe_id)
        if mat is None or len(mat) < 2:
            return None
        return np.diff(mat, axis=0)

    # ── Similarity Queries ────────────────────────────────────────────────

    def query(self, probe: np.ndarray, top_k: int = 3,
              exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Find the top-k most similar lobe states to a probe vector.

        Returns:
            List of (lobe_id, cosine_similarity) sorted descending.
        """
        probe = self._coerce(probe)
        probe_norm = np.linalg.norm(probe)
        if probe_norm < 1e-12:
            return []

        exclude = set(exclude or [])
        results = []
        for lid, vec in self._state.items():
            if lid in exclude:
                continue
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-12:
                continue
            sim = float(np.dot(probe, vec) / (probe_norm * vec_norm))
            results.append((lid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def resonance_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Compute pairwise cosine similarity across all active lobes.

        Returns:
            (lobe_ids, sim_matrix) where sim_matrix[i,j] is cosine
            similarity between lobe_ids[i] and lobe_ids[j].
        """
        ids = list(self._state.keys())
        n = len(ids)
        if n == 0:
            return ids, np.zeros((0, 0))

        mat = np.stack([self._state[lid] for lid in ids])  # (n, dim)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = mat / norms
        sim = normed @ normed.T  # (n, n)
        return ids, sim

    # ── Entanglement (cross-lobe blending) ────────────────────────────────

    def entangle(self, lobe_ids: List[str],
                 weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Blend multiple lobe states into a single entangled vector.

        Args:
            lobe_ids: Lobes to blend.
            weights:  Optional weight per lobe.  Defaults to uniform.

        Returns:
            Weighted sum of lobe state vectors, L2-normalized.
        """
        vecs = []
        for lid in lobe_ids:
            v = self._state.get(lid)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return np.zeros(self.dim)

        stack = np.stack(vecs)  # (k, dim)
        if weights is None:
            weights = np.ones(len(vecs)) / len(vecs)
        else:
            weights = np.asarray(weights[:len(vecs)], dtype=np.float64)
            weights /= (weights.sum() + 1e-12)

        blended = weights @ stack  # (dim,)
        norm = np.linalg.norm(blended)
        if norm > 1e-12:
            blended /= norm
        return blended

    # ── Listener (async event stream) ─────────────────────────────────────

    def subscribe(self, maxsize: int = 64) -> asyncio.Queue:
        """Subscribe to hub write events.  Returns an asyncio.Queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._listeners.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a listener queue."""
        try:
            self._listeners.remove(q)
        except ValueError:
            pass

    # ── Internals ─────────────────────────────────────────────────────────

    def _coerce(self, vector: np.ndarray) -> np.ndarray:
        """Coerce a vector to the hub's fixed dimension."""
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.shape[0] == self.dim:
            return v
        out = np.zeros(self.dim, dtype=np.float64)
        n = min(v.shape[0], self.dim)
        out[:n] = v[:n]
        return out

    # ── Disk Persistence ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save hub state to disk for crash recovery.

        Creates:
          {path}/akashic_state.npz   — all lobe vectors
          {path}/akashic_meta.json   — metadata + timestamps
        """
        import json as _json
        os.makedirs(path, exist_ok=True)

        # Save vectors
        if self._state:
            np.savez_compressed(
                os.path.join(path, "akashic_state.npz"),
                **{lid: vec for lid, vec in self._state.items()}
            )

        # Save metadata + timestamps
        meta = {
            "dim": self.dim,
            "trace_depth": self.trace_depth,
            "timestamps": self._timestamps,
            "meta": {lid: dict(m) for lid, m in self._meta.items()},
            "saved_at": time.time(),
        }
        with open(os.path.join(path, "akashic_meta.json"), "w") as f:
            _json.dump(meta, f, indent=2, default=str)

    def load(self, path: str) -> int:
        """Restore hub state from disk.

        Returns:
            Number of lobes restored.
        """
        import json as _json

        npz_path = os.path.join(path, "akashic_state.npz")
        meta_path = os.path.join(path, "akashic_meta.json")

        count = 0
        if os.path.isfile(npz_path):
            data = np.load(npz_path)
            for lid in data.files:
                vec = data[lid]
                self._state[lid] = self._coerce(vec)
                count += 1

        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = _json.load(f)
            self._timestamps.update(meta.get("timestamps", {}))
            for lid, m in meta.get("meta", {}).items():
                self._meta[lid].update(m)

        return count

    # ── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the hub."""
        return {
            "dim": self.dim,
            "trace_depth": self.trace_depth,
            "active_lobes": len(self._state),
            "lobe_ids": list(self._state.keys()),
            "listeners": len(self._listeners),
            "ages": {lid: self.age(lid) for lid in self._state},
        }

    def __repr__(self):
        return (f"<AkashicHub dim={self.dim} lobes={list(self._state.keys())} "
                f"listeners={len(self._listeners)}>")
