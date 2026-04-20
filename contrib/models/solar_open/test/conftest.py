"""Shared pytest fixtures for Solar Open MoE tests.

Provides session-scoped fixtures for integration tests:
- model_dir: tiny random checkpoint created via SolarOpenForCausalLM.save_pretrained()
- traced_dir: temp directory for compiled Neuron model
- compiled_model: NeuronSolarOpenForCausalLM compiled once per test session
- neuron_config: MoENeuronConfig for the integration tests
"""

import sys
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
# 3. hf_adapter.prepare_inputs_for_generation previously referenced
#    `tensor_capture_hook` which was never assigned.  This has been fixed
#    upstream (the offending line was removed from hf_adapter.py).
#    A minimal safety shim is retained for backward compatibility.
#
# All shims are applied at conftest collection time and do not affect
# Solar Open inference behaviour.
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
    # Renamed to GenerateDecoderOnlyOutput in transformers 5.0
    _tg.SampleDecoderOnlyOutput = _tg.GenerateDecoderOnlyOutput  # type: ignore[attr-defined]
if not hasattr(_tg, "SampleEncoderDecoderOutput"):
    _tg.SampleEncoderDecoderOutput = _tg.GenerateEncoderDecoderOutput  # type: ignore[attr-defined]

# Shim 3 (RESOLVED): The tensor_capture_hook NameError in
# hf_adapter.prepare_inputs_for_generation has been fixed upstream by removing
# the offending line from the model_inputs dict.  The shim below is retained as
# a no-op safety net for running against unpatched NxDI versions.
import neuronx_distributed_inference.utils.hf_adapter as _hfa_mod  # noqa: E402

if not hasattr(_hfa_mod, "tensor_capture_hook"):
    _hfa_mod.tensor_capture_hook = None  # type: ignore[attr-defined]

# Ensure contrib src is on path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Create a temporary tiny random Solar Open model directory.

    Uses SolarOpenForCausalLM(config).save_pretrained() which writes
    config.json, model.safetensors, and generation_config.json automatically.
    """
    from test.integration.utils import create_tiny_solar_open_model

    tmpdir = tmp_path_factory.mktemp("solar_open_model")
    create_tiny_solar_open_model(str(tmpdir))
    return str(tmpdir)


@pytest.fixture(scope="session")
def traced_dir(tmp_path_factory):
    """Temporary directory for the compiled Neuron model."""
    return str(tmp_path_factory.mktemp("solar_open_traced"))


@pytest.fixture(scope="session")
def neuron_config():
    """MoENeuronConfig for integration tests."""
    from test.integration.utils import get_neuron_config

    return get_neuron_config()


@pytest.fixture(scope="session")
def compiled_model(model_dir, traced_dir, neuron_config):
    """Compile NeuronSolarOpenForCausalLM from tiny random checkpoint.

    Skips if Neuron hardware (NeuronCores) is not available.
    Compiles once per test session and returns the loaded model.
    """
    try:
        from solar_open.modeling_solar_open import (
            NeuronSolarOpenForCausalLM,
            SolarOpenInferenceConfig,
            load_solar_open_config,
        )
    except ImportError as e:
        pytest.skip(f"solar_open package not importable: {e}")

    try:
        import torch_neuronx  # noqa: F401
    except ImportError:
        pytest.skip("torch_neuronx not available — Neuron hardware required")

    # Compile
    config = SolarOpenInferenceConfig(
        neuron_config,
        load_config=load_solar_open_config(model_dir),
    )
    model = NeuronSolarOpenForCausalLM(model_dir, config)
    model.compile(traced_dir)

    # Copy model weights to traced_dir so load() can find the safetensors checkpoint.
    # generation_config.json is already written by save_pretrained() in model_dir;
    # copy it so HuggingFaceGenerationAdapter can load it from traced_dir.
    import shutil
    import os

    for fname in ("model.safetensors", "generation_config.json"):
        src = os.path.join(model_dir, fname)
        dst = os.path.join(traced_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Load compiled model
    model = NeuronSolarOpenForCausalLM(traced_dir)
    model.load(traced_dir)
    return model
