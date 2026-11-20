"""Differentiable quantum circuit simulator for SOMA U3 research.

Mathematically genuine unitary statevector evolution, computed CLASSICALLY.
Not connected to real qubits, physical quantum sensing, or IBM Quantum.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class VariationalQuantumFeatures(nn.Module):
    """Trainable RY/RZ/CZ circuit returning Z expectations [batch, qubits]."""

    def __init__(self, input_dim: int, qubits: int = 4, layers: int = 2):
        super().__init__()
        if not 2 <= qubits <= 8 or layers < 1:
            raise ValueError("research simulator supports 2..8 qubits, layers >= 1")
        self.qubits = qubits
        self.layers = layers
        self.encode = nn.Linear(input_dim, qubits)
        self.theta_y = nn.Parameter(0.1 * torch.randn(layers, qubits))
        self.theta_z = nn.Parameter(0.1 * torch.randn(layers, qubits))
        dimensions = torch.arange(2**qubits)
        signs = []
        for i in range(qubits):
            bit = (dimensions >> (qubits - i - 1)) & 1
            signs.append(1 - 2 * bit)
        self.register_buffer("z_signs", torch.stack(signs).float(), persistent=False)
        phase = torch.ones(2**qubits)
        for i in range(qubits - 1):
            both = ((dimensions >> (qubits - i - 1)) & 1) * ((dimensions >> (qubits - i - 2)) & 1)
            phase = phase * (1 - 2 * both)
        self.register_buffer("cz_phase", phase, persistent=False)

    def _rotate(self, states: torch.Tensor, theta: torch.Tensor, qubit: int, axis: str):
        b = states.shape[0]
        if axis == "y":
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            gate = torch.stack((torch.stack((c, -s)), torch.stack((s, c)))).to(states.dtype)
        else:
            gate = torch.diag(torch.stack((torch.exp(-0.5j * theta), torch.exp(0.5j * theta))))
        shaped = states.reshape(b, *([2] * self.qubits))
        moved = shaped.movedim(1 + qubit, -1)
        return (moved @ gate.T).movedim(-1, 1 + qubit).reshape(b, -1)

    def _rotate_batch(self, states: torch.Tensor, theta: torch.Tensor, qubit: int):
        """Per-example RY input encoding."""
        b = states.shape[0]
        a = states.reshape(b, *([2] * self.qubits)).movedim(1 + qubit, -1)
        c, s = torch.cos(theta / 2), torch.sin(theta / 2)
        expanded = (b,) + (1,) * (self.qubits - 1)
        c, s = c.reshape(expanded), s.reshape(expanded)
        out = torch.stack((c * a[..., 0] - s * a[..., 1],
                           s * a[..., 0] + c * a[..., 1]), dim=-1)
        return out.movedim(-1, 1 + qubit).reshape(b, -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("quantum input must be [batch, features]")
        if features.dtype not in (torch.float32, torch.float64):
            features = features.float()
        angle = math.pi * torch.tanh(self.encode(features))
        dtype = torch.complex128 if features.dtype == torch.float64 else torch.complex64
        state = torch.zeros(features.shape[0], 2**self.qubits,
                            device=features.device, dtype=dtype)
        state[:, 0] = 1.0
        for i in range(self.qubits):
            state = self._rotate_batch(state, angle[:, i], i)
        for layer in range(self.layers):
            for i in range(self.qubits):
                state = self._rotate(state, self.theta_y[layer, i], i, "y")
                state = self._rotate(state, self.theta_z[layer, i], i, "z")
            state = state * self.cz_phase.to(device=state.device, dtype=state.dtype)
        probs = (state.conj() * state).real
        return probs @ self.z_signs.to(device=probs.device, dtype=probs.dtype).T
