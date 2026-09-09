"""Regression tests for episode-boundary action masking."""

import json
import sys
import types

import numpy as np
import pytest
import torch

from opendm.data.collator import NormStatsCollator
from opendm.model.dm05.dm05_utils import masked_flow_matching_loss

# The transform module only needs TransformPipeline for image transforms.  Keep
# these tests runnable in a minimal CPU checkout where optional augmentation
# dependencies are not installed.
if "opendm.data.augmentations" not in sys.modules:
    augmentations_stub = types.ModuleType("opendm.data.augmentations")
    augmentations_stub.TransformPipeline = object
    sys.modules["opendm.data.augmentations"] = augmentations_stub

from opendm.data.transforms import BuildActionChunk  # noqa: E402


def _episode(states, actions=None):
    lines = []
    for index, state in enumerate(states):
        frame = {"state": list(state)}
        if actions is not None:
            frame["action"] = list(actions[index])
        lines.append(json.dumps(frame))
    return lines


def test_chunk_marks_repeated_terminal_tail_invalid():
    transform = BuildActionChunk(action_horizon=4)
    data = {
        "state": np.asarray([1.0, 1.0], dtype=np.float32),
        "raw_lines": _episode([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        "meta_data": {"frame_index": 1},
    }

    result = transform(data)

    np.testing.assert_allclose(
        result["action"],
        [[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]],
    )
    np.testing.assert_array_equal(
        result["action_mask"],
        [[[True, True], [False, False], [False, False], [False, False]]],
    )


def test_explicit_actions_keep_validity_and_values():
    transform = BuildActionChunk(action_horizon=4)
    data = {
        "state": np.asarray([0.0], dtype=np.float32),
        "action": np.asarray([10.0], dtype=np.float32),
        "raw_lines": _episode(
            [[0.0], [1.0], [2.0]],
            actions=[[10.0], [20.0], [30.0]],
        ),
        "meta_data": {"frame_index": 1},
    }

    result = transform(data)

    np.testing.assert_allclose(result["action"], [[[20.0], [30.0], [30.0], [30.0]]])
    np.testing.assert_array_equal(
        result["action_mask"],
        [[[True], [True], [False], [False]]],
    )


def test_full_window_is_deterministic_and_fully_valid():
    transform = BuildActionChunk(action_horizon=2)
    data = {
        "state": np.asarray([0.0], dtype=np.float32),
        "raw_lines": _episode([[0.0], [1.0], [2.0], [3.0]]),
        "meta_data": {"frame_index": 1},
    }

    first = transform({**data, "meta_data": dict(data["meta_data"])})
    second = transform({**data, "meta_data": dict(data["meta_data"])})

    np.testing.assert_array_equal(first["action"], [[[2.0], [3.0]]])
    np.testing.assert_array_equal(first["action_mask"], [[[True], [True]]])
    np.testing.assert_array_equal(first["action"], second["action"])
    np.testing.assert_array_equal(first["action_mask"], second["action_mask"])


def test_norm_stats_collator_excludes_invalid_action_rows():
    collator = NormStatsCollator()
    instances = [
        {
            "meta_data": {"robot_type": "arm"},
            "state": np.asarray([0.0, 0.0], dtype=np.float32),
            "action": np.asarray([[[1.0, 10.0], [2.0, 20.0], [99.0, 99.0]]]),
            "action_mask": np.asarray(
                [[[True, True], [True, True], [False, False]]],
            ),
        },
        {
            "meta_data": {"robot_type": "arm"},
            "state": np.asarray([0.0, 0.0], dtype=np.float32),
            "action": np.asarray([[[3.0, 30.0], [88.0, 88.0], [77.0, 77.0]]]),
            "action_mask": np.asarray(
                [[[True, True], [False, False], [False, False]]],
            ),
        },
    ]

    result = collator(instances)["robot_batches"]["arm"]["action"]

    np.testing.assert_array_equal(
        result,
        np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
    )


def test_norm_stats_collator_rejects_dimension_specific_mask():
    instance = {
        "meta_data": {"robot_type": "arm"},
        "state": np.asarray([0.0], dtype=np.float32),
        "action": np.asarray([[[1.0, 2.0]]]),
        "action_mask": np.asarray([[[True, False]]]),
    }

    with pytest.raises(ValueError, match="uniform across action dimensions"):
        NormStatsCollator()([instance])


def test_masked_flow_loss_matches_unmasked_mse_for_full_mask():
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)

    loss = masked_flow_matching_loss(prediction, target, mask)

    torch.testing.assert_close(loss, torch.nn.functional.mse_loss(prediction, target).view(1))


def test_masked_flow_loss_normalizes_each_sample_and_handles_empty_rows():
    prediction = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[5.0, 7.0], [11.0, 13.0]],
        ],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    # Singleton feature axis exercises broadcast expansion; the second sample
    # is fully padded and must contribute zero without NaN or gradients.
    mask = torch.tensor([[[True], [False]], [[False], [False]]])

    loss = masked_flow_matching_loss(prediction, target, mask)
    loss.mean().backward()

    torch.testing.assert_close(loss, torch.tensor([2.0, 0.0]))
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(prediction.grad[0], torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    torch.testing.assert_close(prediction.grad[1], torch.zeros_like(prediction.grad[1]))
