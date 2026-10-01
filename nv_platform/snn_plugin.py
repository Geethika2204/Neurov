"""
/platform/snn_plugin.py

Layer 1 — SNN Input Plugin
===========================
Converts a raw sensor stream (normalised float tensor) into a spike tensor
using a spiking neural network.

Interface
---------
    plugin = SNNInputPlugin(backend="snntorch", num_steps=25)
    spike_tensor = plugin.process_events(sensor_stream)

    sensor_stream : Tensor  shape (batch, time_window, features)
    spike_tensor  : Tensor  shape (num_steps, batch, features)

Backend swapping
----------------
Pass ``backend="spikingjelly"`` or ``backend="norse"`` to swap the underlying
SNN without changing any other layer or domain code.  Non-default backends
currently raise ``NotImplementedError`` (stubs ready for implementation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------


class _SNNBackend(ABC):
    """Internal protocol every SNN backend must satisfy."""

    @abstractmethod
    def encode(self, sensor_stream: Tensor, num_steps: int) -> Tensor:
        """
        Parameters
        ----------
        sensor_stream : Tensor, shape (batch, time_window, features)
        num_steps     : int — number of SNN simulation time steps

        Returns
        -------
        Tensor, shape (num_steps, batch, features)
        """


# ---------------------------------------------------------------------------
# snnTorch backend (default)
# ---------------------------------------------------------------------------


class _SNNTorchBackend(_SNNBackend):
    """
    snnTorch-based backend using LIF rate coding.

    Encoding pipeline (matches preprocess.py conventions):
        1. Mean-pool across time window → (batch, features)
        2. Sigmoid to map to (0, 1)  — acts as firing-rate proxy
        3. snntorch.spikegen.rate → (num_steps, batch, features)
    """

    def __init__(self) -> None:
        try:
            import snntorch  # noqa: F401
            import snntorch.spikegen as spikegen  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "snntorch is required for the default backend. "
                "Install it with: pip install snntorch"
            ) from exc

    def encode(self, sensor_stream: Tensor, num_steps: int) -> Tensor:
        import snntorch.spikegen as spikegen

        # Mean-pool time window: (batch, time_window, features) → (batch, features)
        if sensor_stream.dim() == 3:
            pooled = sensor_stream.mean(dim=1)
        elif sensor_stream.dim() == 2:
            pooled = sensor_stream  # already (batch, features)
        else:
            raise ValueError(
                f"Expected sensor_stream with 2 or 3 dims, got {sensor_stream.dim()}"
            )

        # Sigmoid → (0, 1) firing probability per feature
        firing_prob = torch.sigmoid(pooled)

        # Rate coding: (num_steps, batch, features)
        spikes = spikegen.rate(firing_prob, num_steps=num_steps)
        return spikes


# ---------------------------------------------------------------------------
# Stub backends
# ---------------------------------------------------------------------------


class _SpikingJellyBackend(_SNNBackend):  # pragma: no cover
    """
    SpikingJelly backend stub.

    To implement: replace ``raise NotImplementedError`` with a
    ``spikingjelly.activation_based`` LIF layer encoding the same
    (batch, time_window, features) → (num_steps, batch, features) contract.
    """

    def encode(self, sensor_stream: Tensor, num_steps: int) -> Tensor:
        raise NotImplementedError(
            "SpikingJelly backend is not yet implemented. "
            "Implement _SpikingJellyBackend.encode() in platform/snn_plugin.py."
        )


class _NorseBackend(_SNNBackend):  # pragma: no cover
    """
    Norse backend stub.

    To implement: replace ``raise NotImplementedError`` with a
    ``norse.torch`` LIF layer following the same tensor contract.
    """

    def encode(self, sensor_stream: Tensor, num_steps: int) -> Tensor:
        raise NotImplementedError(
            "Norse backend is not yet implemented. "
            "Implement _NorseBackend.encode() in platform/snn_plugin.py."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[_SNNBackend]] = {
    "snntorch": _SNNTorchBackend,
    "spikingjelly": _SpikingJellyBackend,
    "norse": _NorseBackend,
}


# ---------------------------------------------------------------------------
# Public plugin class
# ---------------------------------------------------------------------------


class SNNInputPlugin:
    """
    Layer 1 of the NeuroVerse pipeline.

    Parameters
    ----------
    backend   : str   Backend name. One of ``"snntorch"`` (default),
                      ``"spikingjelly"``, ``"norse"``.
    num_steps : int   Number of SNN simulation time steps (default 25).

    Examples
    --------
    >>> plugin = SNNInputPlugin()
    >>> spikes = plugin.process_events(torch.randn(4, 20, 12))
    >>> spikes.shape
    torch.Size([25, 4, 12])
    """

    SUPPORTED_BACKENDS: ClassVar[tuple[str, ...]] = tuple(_BACKENDS)

    def __init__(
        self,
        backend: str = "snntorch",
        num_steps: int = 25,
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Choose from: {list(_BACKENDS)}"
            )
        self._num_steps = num_steps
        self._backend: _SNNBackend = _BACKENDS[backend]()
        self._backend_name = backend

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def process_events(self, sensor_stream: Tensor) -> Tensor:
        """
        Encode a sensor stream into a spike tensor.

        Parameters
        ----------
        sensor_stream : Tensor
            Shape ``(batch, time_window, features)`` or ``(batch, features)``.
            Values should be normalised (zero-mean, unit-variance recommended).

        Returns
        -------
        Tensor
            Shape ``(num_steps, batch, features)`` — binary spike trains.
        """
        if not isinstance(sensor_stream, Tensor):
            sensor_stream = torch.tensor(sensor_stream, dtype=torch.float32)
        if sensor_stream.dtype != torch.float32:
            sensor_stream = sensor_stream.float()

        spikes = self._backend.encode(sensor_stream, self._num_steps)

        # Validate output shape contract
        expected_features = (
            sensor_stream.shape[-1]
        )
        if spikes.shape[0] != self._num_steps:
            raise RuntimeError(
                f"Backend {self._backend_name!r} returned spike tensor with "
                f"{spikes.shape[0]} steps; expected {self._num_steps}."
            )
        if spikes.shape[-1] != expected_features:
            raise RuntimeError(
                f"Backend {self._backend_name!r} changed feature dim from "
                f"{expected_features} to {spikes.shape[-1]}."
            )
        return spikes
