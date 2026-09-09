"""CPU regression coverage for the suffix graph's time-conditioning call."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from opendm.constants.precision import (
    BF16_MIXED_PRECISION_POLICY,
    FP32_MIXED_PRECISION_POLICY,
)
from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration


def _tiny_conditioner(dtype, precision_policy=BF16_MIXED_PRECISION_POLICY):
    """Bind the production method without constructing a VLM or action expert."""
    hidden_size = 8
    layers = [
        torch.nn.Linear(hidden_size, hidden_size, device="cpu", dtype=dtype)
        for _ in range(2)
    ]
    with torch.no_grad():
        for layer in layers:
            layer.weight.fill_(0.125)
            layer.bias.fill_(0.0625)
    model = SimpleNamespace(
        action_in_proj=SimpleNamespace(out_features=hidden_size),
        time_mlp_in=layers[0],
        time_mlp_out=layers[1],
    )
    owner = SimpleNamespace(model=model, precision_policy=precision_policy)
    owner._build_adarms_cond = MethodType(
        DM05ForConditionalGeneration._build_adarms_cond, owner
    )
    return owner


def test_build_adarms_cond_accepts_explicit_suffix_dtype():
    model = _tiny_conditioner(torch.float32)
    time = torch.tensor([0.2, 0.8], device="cpu")
    input_dtypes = []
    hook = model.model.time_mlp_in.register_forward_pre_hook(
        lambda module, inputs: input_dtypes.append(inputs[0].dtype)
    )
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # _run_suffix_graph_step passes the suffix dtype positionally.
            actual = model._build_adarms_cond(time, torch.bfloat16)
    finally:
        hook.remove()

    assert input_dtypes == [torch.bfloat16]
    assert actual.shape == (2, 8)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_build_adarms_cond_legacy_call_uses_weight_dtype(weight_dtype):
    model = _tiny_conditioner(weight_dtype)
    time = torch.tensor([0.2, 0.8], device="cpu")

    actual = model._build_adarms_cond(time)
    explicit_default = model._build_adarms_cond(time, dtype=None)

    assert actual.shape == (2, 8)
    assert actual.dtype == weight_dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, explicit_default, rtol=0, atol=0)


def test_build_adarms_cond_fp32_mixed_preserves_fp32_under_autocast():
    model = _tiny_conditioner(torch.float32, FP32_MIXED_PRECISION_POLICY)
    time = torch.tensor([0.2, 0.8], device="cpu")
    reference = model._build_adarms_cond(time)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = model._build_adarms_cond(time, torch.bfloat16)

    assert actual.shape == (2, 8)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
