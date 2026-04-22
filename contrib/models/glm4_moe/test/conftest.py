# coding=utf-8
"""Shared pytest fixtures for GLM-4.5 MoE tests."""

import random
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Compatibility shims for transformers 5.0.0 + NxDI library quirks
#
# Three issues are resolved here before any test-module import occurs:
#
# 1. neuronx_distributed.pipeline.trace imports transformers.utils.fx.HFTracer
#    which was removed in transformers 5.0.  Register a stub module BEFORE
#    neuronx_distributed is first imported so the import succeeds.
#
# 2. neuronx_distributed_inference.utils.hf_adapter imports
#    transformers.generation.SampleDecoderOnlyOutput which was renamed to
#    GenerateDecoderOnlyOutput in transformers 5.0.  Patch the live
#    transformers.generation module to re-export the old name as an alias.
#
# 3. hf_adapter.prepare_inputs_for_generation references tensor_capture_hook
#    which was never assigned.  The direct fix is to remove the line from
#    hf_adapter.py; this shim is a safety net for unpatched NxDI versions.
#
# All shims are applied at conftest collection time and do not affect
# GLM-4.5 MoE inference behaviour.
# ---------------------------------------------------------------------------

# Shim 1: transformers.utils.fx.HFTracer stub
if "transformers.utils.fx" not in sys.modules:
    _fx_stub = types.ModuleType("transformers.utils.fx")

    class _HFTracerStub:
        """Stub replacing transformers.utils.fx.HFTracer (removed in transformers 5.0)."""

    _fx_stub.HFTracer = _HFTracerStub  # type: ignore[attr-defined]
    sys.modules["transformers.utils.fx"] = _fx_stub

# Shim 2: transformers.generation.SampleDecoderOnlyOutput backward-compat alias
import transformers.generation as _tg

if not hasattr(_tg, "SampleDecoderOnlyOutput"):
    _tg.SampleDecoderOnlyOutput = _tg.GenerateDecoderOnlyOutput  # type: ignore[attr-defined]
if not hasattr(_tg, "SampleEncoderDecoderOutput"):
    _tg.SampleEncoderDecoderOutput = _tg.GenerateEncoderDecoderOutput  # type: ignore[attr-defined]

# Shim 3: tensor_capture_hook safety net for unpatched hf_adapter
import neuronx_distributed_inference.utils.hf_adapter as _hfa_mod  # noqa: E402

if not hasattr(_hfa_mod, "tensor_capture_hook"):
    _hfa_mod.tensor_capture_hook = None  # type: ignore[attr-defined]

# Add src to path so glm4_moe package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session", autouse=True)
def random_seed():
    """Fix all random seeds for reproducibility."""
    torch.manual_seed(42)
    random.seed(42)
    try:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(42)
    except ImportError:
        pass
    yield


@pytest.fixture(scope="session")
def glm4_moe_config():
    """Load GLM-4.5 MoE test config (full architecture, few layers)."""
    import json

    config_path = Path(__file__).parent / "integration" / "config_glm4_moe_2layers.json"
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def tmp_dir_path():
    """Create a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
