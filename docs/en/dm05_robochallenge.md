# DM05 RoboChallenge Table 30 v2 Inference Guide

This document describes how to run the DM05 RoboChallenge Table 30 v2 inference
client provided under `third_party/robochallenge_inference`.

## Overview

The RoboChallenge inference client connects to the RoboChallenge platform,
selects the active job from a submission, fetches robot observations, runs a
DM05 policy, and submits actions back to the platform.

The client supports four Table 30 v2 robot configs:

- `arx5`
- `ur5`
- `aloha`
- `w1`

## Files

```text
third_party/robochallenge_inference/
├── configs/
│   ├── default.yaml
│   └── generalist/      # arx5 / ur5 / aloha / w1 configs
├── policies/            # OpenDM-backed DM05 policy and output processing
├── robot/               # RoboChallenge HTTP client and job loop
├── runner/              # Policy invocation and debug capture
├── utils/               # Task metadata, transforms, logging, helpers
├── execute.py
└── requirements.txt
```

## Environment

Install OpenDM first. Fast inference requires the `fast-infer` optional
dependencies:

```bash
# Run from the OpenDM repository root.
pip install -e ".[fast-infer]"

cd third_party/robochallenge_inference
pip install -r requirements.txt
```

Set model paths before running inference:

```bash
export OPENDM_ROOT=/path/to/opendm

export ARX5_CHECKPOINT=/path/to/arx5/checkpoint
export ARX5_NORM_STATS=/path/to/arx5/norm_stats.json

export UR5_CHECKPOINT=/path/to/ur5/checkpoint
export UR5_NORM_STATS=/path/to/ur5/norm_stats.json

export ALOHA_CHECKPOINT=/path/to/aloha/checkpoint
export ALOHA_NORM_STATS=/path/to/aloha/norm_stats.json

export W1_CHECKPOINT=/path/to/w1/checkpoint
export W1_NORM_STATS=/path/to/w1/norm_stats.json
```

If `*_NORM_STATS` is omitted, the client falls back to
`CHECKPOINT/norm_stats.json`.

## Run Inference

Run the client from `third_party/robochallenge_inference`:

```bash
cd third_party/robochallenge_inference

python execute.py \
  --config-name generalist/arx5 \
  user_id=YOUR_USER_ID \
  submission_id=YOUR_SUBMISSION_ID \
  run_id=YOUR_RUN_ID
```

Use the matching robot config for each run:

```bash
python execute.py --config-name generalist/arx5  user_id=YOUR_USER_ID submission_id=YOUR_SUBMISSION_ID run_id=YOUR_RUN_ID
python execute.py --config-name generalist/ur5   user_id=YOUR_USER_ID submission_id=YOUR_SUBMISSION_ID run_id=YOUR_RUN_ID
python execute.py --config-name generalist/aloha user_id=YOUR_USER_ID submission_id=YOUR_SUBMISSION_ID run_id=YOUR_RUN_ID
python execute.py --config-name generalist/w1    user_id=YOUR_USER_ID submission_id=YOUR_SUBMISSION_ID run_id=YOUR_RUN_ID
```

`run_id` is optional. When omitted, the worker polls the submission and selects
matching active jobs for the selected robot type.

## Launch Arguments

The launcher uses Hydra overrides.

| Argument | Required | Purpose |
| --- | --- | --- |
| `--config-name generalist/<robot>` | yes | Select `arx5`, `ur5`, `aloha`, or `w1` |
| `user_id=...` | yes | RoboChallenge user id used for platform requests |
| `submission_id=...` | yes | Submission id containing the run collection |
| `run_id=...` | optional | Limit execution to one run in the submission |
| `checkpoint=...` | optional | Override checkpoint path from environment variables |
| `norm_stats=...` | optional | Override norm statistics path |
| `action_horizon=...` | optional | Override action horizon |
| `action_playback_target_steps=...` | optional | Uniformly sample generated actions to a target count |
| `debug=true` | optional | Enable per-step debug capture |
| `debug_image_limit=...` | optional | Number of platform image snapshots to save when debug is enabled; negative values save all snapshots |
| `log_dir=...` | optional | Runtime log and optional debug-capture directory |
| `hydra.run.dir=...` | optional | Hydra output directory |

By default, debug capture is disabled to avoid large log directories. Runtime
logs are still written under `log_dir`. To save per-step replay data and a
bounded number of platform image snapshots, pass:

```bash
python execute.py \
  --config-name generalist/arx5 \
  user_id=YOUR_USER_ID \
  submission_id=YOUR_SUBMISSION_ID \
  run_id=YOUR_RUN_ID \
  debug=true \
  debug_image_limit=20
```

Use `debug_image_limit=-1` only when full image capture is required.

## Fast Backend Defaults

Fast inference defaults live in
`configs/default.yaml` under `robot_profiles.<robot>.runtime_args`.

Configured TensorRT engine paths:

| Robot | Default engine path |
| --- | --- |
| ARX5 | `checkpoints/trt_engines/dm05_arx5_h8.engine` |
| UR5 | `checkpoints/trt_engines/dm05_ur5_h2.engine` |
| ALOHA | `checkpoints/trt_engines/dm05_aloha_h3.engine` |
| W1 | `checkpoints/trt_engines/dm05_w1_h3.engine` |

The suffix indicates the TensorRT vision engine image count:

- `h8`: ARX5 uses 3 current images plus 5 history slots.
- `h2`: UR5 uses 2 current images.
- `h3`: ALOHA and W1 use 3 current images.

If an engine does not exist, OpenDM builds it on the first fast-backend startup.
To force rebuilding:

```bash
python execute.py \
  --config-name generalist/arx5 \
  user_id=YOUR_USER_ID \
  submission_id=YOUR_SUBMISSION_ID \
  run_id=YOUR_RUN_ID \
  +runtime_args.force_rebuild_trt=true
```

To disable the fast backend for a run:

```bash
python execute.py \
  --config-name generalist/arx5 \
  user_id=YOUR_USER_ID \
  submission_id=YOUR_SUBMISSION_ID \
  run_id=YOUR_RUN_ID \
  runtime_args.backend=default
```

## Runtime Defaults

Robot-level defaults live in `configs/default.yaml`.

- ARX5 uses logical-step history with `action_horizon=50` and
  `action_playback_target_steps=25`.
- UR5 uses `action_horizon=25` and disables playback sampling.
- ALOHA and W1 default to `action_horizon=25`.
- Per-task overrides are defined in `configs/generalist/*.yaml`.
