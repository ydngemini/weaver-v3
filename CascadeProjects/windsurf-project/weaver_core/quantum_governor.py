"""
quantum_governor.py — Adaptive Temporal Governor for QPU Sessions

Manages Qiskit Runtime sessions with a hard 600s time-box and an
adaptive threshold that scales with call density to prevent API
rate-limiting during entropy spikes.

Equations:
    T_max = 600 (QPU session cap in seconds)
    tau_t = tau_base + lambda * (N_calls / delta_t)

Target backend: ibm_miami (Nighthawk 120-qubit)
"""

import time
from dataclasses import dataclass, field

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2


@dataclass
class GovernorMetrics:
    n_calls: int = 0
    session_start: float = 0.0
    last_call_time: float = 0.0
    current_tau: float = 0.5
    throttled_count: int = 0


class QuantumGovernor:
    T_MAX: int = 600
    BACKEND: str = "ibm_miami"

    def __init__(
        self,
        tau_base: float = 0.5,
        lambda_rate: float = 0.01,
        channel: str = "ibm_quantum",
    ):
        self.tau_base = tau_base
        self.lambda_rate = lambda_rate
        self.channel = channel
        self.metrics = GovernorMetrics()
        self._service: QiskitRuntimeService | None = None
        self._session: Session | None = None

    def connect(self) -> "QuantumGovernor":
        self._service = QiskitRuntimeService(channel=self.channel)
        return self

    def open_session(self) -> Session:
        if self._service is None:
            self.connect()
        self._session = Session(
            backend=self.BACKEND,
            max_time=self.T_MAX,
        )
        self.metrics.session_start = time.time()
        self.metrics.n_calls = 0
        return self._session

    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def elapsed(self) -> float:
        if self.metrics.session_start == 0:
            return 0.0
        return time.time() - self.metrics.session_start

    @property
    def remaining(self) -> float:
        return max(0.0, self.T_MAX - self.elapsed)

    def adaptive_threshold(self) -> float:
        delta_t = self.elapsed or 1.0
        tau_t = self.tau_base + self.lambda_rate * (self.metrics.n_calls / delta_t)
        self.metrics.current_tau = tau_t
        return tau_t

    def should_throttle(self, entropy: float) -> bool:
        tau = self.adaptive_threshold()
        if entropy > tau:
            self.metrics.throttled_count += 1
            return True
        return False

    def record_call(self):
        self.metrics.n_calls += 1
        self.metrics.last_call_time = time.time()

    def execute_circuit(self, circuit, shots: int = 4096):
        if self._session is None:
            raise RuntimeError("No active session — call open_session() first")
        if self.remaining <= 0:
            raise TimeoutError(
                f"QPU time-box exceeded: {self.elapsed:.1f}s >= {self.T_MAX}s"
            )
        self.record_call()
        sampler = SamplerV2(session=self._session)
        job = sampler.run([circuit], shots=shots)
        return job.result()

    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def status(self) -> dict:
        return {
            "backend": self.BACKEND,
            "t_max": self.T_MAX,
            "elapsed_s": round(self.elapsed, 1),
            "remaining_s": round(self.remaining, 1),
            "n_calls": self.metrics.n_calls,
            "tau_current": round(self.metrics.current_tau, 4),
            "throttled": self.metrics.throttled_count,
        }

    def __enter__(self):
        self.open_session()
        return self

    def __exit__(self, *_):
        self.close()
