"""tests/test_snn_plugin.py — Unit tests for SNNInputPlugin."""

import pytest
import torch
from nv_platform.snn_plugin import SNNInputPlugin


def test_default_backend_name():
    plugin = SNNInputPlugin()
    assert plugin.backend_name == "snntorch"


def test_output_shape_3d_input():
    """(batch, time_window, features) → (num_steps, batch, features)."""
    plugin = SNNInputPlugin(num_steps=25)
    stream = torch.randn(4, 20, 12)
    spikes = plugin.process_events(stream)
    assert spikes.shape == (25, 4, 12), f"Got {spikes.shape}"


def test_output_shape_2d_input():
    """(batch, features) → (num_steps, batch, features)."""
    plugin = SNNInputPlugin(num_steps=10)
    stream = torch.randn(8, 12)
    spikes = plugin.process_events(stream)
    assert spikes.shape == (10, 8, 12)


def test_spikes_are_binary():
    """Output should contain only 0.0 and 1.0 (rate-coded spikes)."""
    plugin = SNNInputPlugin(num_steps=25)
    stream = torch.randn(4, 20, 12)
    spikes = plugin.process_events(stream)
    unique = spikes.unique()
    assert set(unique.tolist()).issubset({0.0, 1.0}), (
        f"Non-binary values in spikes: {unique}"
    )


def test_custom_num_steps():
    plugin = SNNInputPlugin(num_steps=50)
    stream = torch.randn(2, 10, 6)
    spikes = plugin.process_events(stream)
    assert spikes.shape[0] == 50


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        SNNInputPlugin(backend="nonexistent")


def test_stub_backend_raises_not_implemented():
    """SpikingJelly and Norse stubs raise NotImplementedError."""
    plugin = SNNInputPlugin(backend="spikingjelly")
    with pytest.raises(NotImplementedError):
        plugin.process_events(torch.randn(2, 5, 4))


def test_backend_swap_preserves_shape():
    """
    The shape contract (num_steps, batch, features) is identical across backends.
    Using a custom backend stub to verify without needing SpikingJelly installed.
    """
    from nv_platform.snn_plugin import _BACKENDS, _SNNBackend

    class _StubBackend(_SNNBackend):
        def encode(self, stream, num_steps):
            # Produce valid binary spikes from scratch
            batch = stream.shape[0]
            features = stream.shape[-1]
            return torch.randint(0, 2, (num_steps, batch, features)).float()

    # Temporarily register the stub
    _BACKENDS["_stub"] = _StubBackend
    try:
        plugin = SNNInputPlugin(backend="_stub", num_steps=15)
        stream = torch.randn(3, 10, 7)
        spikes = plugin.process_events(stream)
        assert spikes.shape == (15, 3, 7)
    finally:
        del _BACKENDS["_stub"]


def test_float_conversion():
    """Integer tensors are automatically cast to float32."""
    plugin = SNNInputPlugin(num_steps=5)
    stream = torch.randint(0, 10, (2, 5, 4))
    spikes = plugin.process_events(stream)
    assert spikes.dtype == torch.float32
