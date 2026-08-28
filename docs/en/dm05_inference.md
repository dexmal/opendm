# DM05 Inference Guide

This is the canonical guide for starting the DM05 inference service and calling
its HTTP API. Benchmark and training guides link here instead of duplicating
inference setup.

[中文](../zh/dm05_inference.md)

## 1. Before You Start

Run all commands from the OpenDM repository root. Prepare:

- An OpenDM environment installed according to the main README.
- If you will use `--inference-config.backend fast`, install the additional
  dependency layer in that same environment with `pip install -e ".[fast-infer]"`.
- A DM05 checkpoint compatible with the selected playground entry point.
- The matching `norm_stats.json` in the checkpoint directory. If it is absent,
  OpenDM falls back to the matching file under `./norm_stats/`.
- One NVIDIA GPU.

One `norm_stats.json` may contain multiple robot profiles under
`norm_stats_by_robot`. OpenDM selects the exact profile using
`observation.robot_type`; if it is omitted, the configured dataset default or
the file's `default_robot_type` is used. Historical single-profile files remain
supported. An unknown robot never falls back silently to the default profile.

The checkpoint, playground entry point, `chunk_size`, image keys, state/action
dimensions, and normalization statistics must come from the same training
configuration.

For the fast backend, also keep these runtime prerequisites aligned with the
code path:

- TensorRT is required, not optional. Fast startup builds or loads the vision
  TensorRT engine before serving requests.
- Triton is required, not optional. The fast suffix big-kernel path and the
  fast prefix decoder path both import and dispatch Triton kernels.
- PyTorch FlexAttention support is required. Fast inference forces the LLM
  attention backend to `flex_attention`, and the static prefix fastpath checks
  every decoder layer for that backend.
- Use a PyTorch build that provides
  `torch.nn.attention.flex_attention` (for example `torch>=2.5`).

### Fast Backend Preflight Checklist

Before the first fast launch, confirm:

- CUDA GPU is visible in the target environment.
- `pip install -e ".[fast-infer]"` has completed in that environment.
- `python -c "import tensorrt"` succeeds.
- `python -c "import triton"` succeeds.
- `python -c "import torch.nn.attention.flex_attention"` succeeds.
- The checkpoint, playground entry point, `chunk_size`, action dimension, and
  `image_prompts` all come from the same training run.
- The uploaded image count and order match `--inference-config.image-prompts`.
- Extra time is reserved for the first startup to export ONNX and build the
  TensorRT engine before the HTTP service becomes ready.

## 2. Choose an Entry Point

| Use case | Entry point | Typical checkpoint | Chunk size | Images | Action dimension |
| --- | --- | --- | ---: | ---: | ---: |
| Base pretrained model | `opendm/exp/dm05_exp.py` | `Dexmal/DM05` | 50 | 3 | 14 |
| LIBERO | `playground/dm05_libero.py` | `Dexmal/DM05-libero` | 10 | 2 | 7 |
| RoboTwin 2.0 | `playground/dm05_robotwin2.py` | `Dexmal/DM05-robotwin2` | 50 | 3 | 14 |
| Demo or custom SFT | `playground/dm05_sft_demo.py` or your own entry | Your SFT checkpoint | Training value | Training value | Training value |
| LIBERO LoRA | `playground/dm05_libero_lora.py` | A LIBERO LoRA step checkpoint | 10 | 2 | 7 |

Do not mix a benchmark checkpoint with another benchmark's entry point or
inference dimensions.

## 3. Start the Default Backend

The default backend uses the standard PyTorch path for the VLM prefix. On CUDA
in evaluation mode, the default `sdpa` action-attention backend automatically
uses CUDA Graph replay for the action suffix; unsupported configurations safely
remain on eager execution. No additional `fast-infer` dependencies are needed.

Suffix Graph profiles are created lazily per execution shape. The first request
in a new profile runs eagerly, the second captures the Graph, and later requests
reuse it. Prefix lengths are rounded up to 16-token buckets, and up to eight
profiles are retained. This supports varying prompts without a fixed Graph
prefix limit; a capture or replay failure falls back to eager execution.

The inference launcher starts one Python process directly, so
`--nproc_per_node` is not needed.

### DM05 Base Pretrained Model

Download the base pretrained checkpoint:

```bash
hf download Dexmal/DM05 \
  --local-dir ./checkpoints/DM05
```

Start the service:

```bash
script/dm05_launcher.sh \
  --exp opendm/exp/dm05_exp.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

The base pretrained checkpoint uses three images and a 14-dimensional
state/action. Set `observation.robot_type` to `DOS W1` or `Aloha` to select the
matching normalization profile. If the field is omitted, the checkpoint's
default profile, `DOS W1`, is used.

Requests to the base pretrained model should explicitly provide
`observation.control_mode` and `observation.speed` so that the input matches
the pretraining conditions. If `observation.speed` is omitted, the service
defaults it to `"0.5"`.

### LIBERO

Download the released checkpoint:

```bash
hf download Dexmal/DM05-libero \
  --local-dir ./checkpoints/DM05-libero
```

Start the service:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --model-config.chunk-size 10 \
  --inference-config.output-action-dim 7 \
  --inference-config.image-prompts "Head" "Left wrist" \
  --inference-config.port 7891
```

The LIBERO entry point uses two images, an 8-dimensional state, a 7-dimensional
action, and the `Franka` robot type.

### RoboTwin 2.0

Download the released checkpoint:

```bash
hf download Dexmal/DM05-robotwin2 \
  --local-dir ./checkpoints/DM05-robotwin2-bf16
```

Start the service:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robotwin2.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-robotwin2-bf16 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

The RoboTwin 2.0 entry point uses three images, a 14-dimensional state/action,
and the `Aloha RoboTwin2` robot type.

### Demo or Custom SFT

Use the checkpoint produced by training and keep all data-dependent settings
consistent with that run. For the built-in demo configuration:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --task inference \
  --data-config.dataset-name demo \
  --model-config.model-name-or-path ./user_checkpoints/dm05_sft_demo_smoke/checkpoint-10 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

For custom data, replace the entry point, dataset name, checkpoint, chunk size,
image keys, and action dimension with the values used during training.

### LIBERO LoRA

Pass a LoRA step checkpoint as `model-name-or-path`. The loader reads its
`adapter_config.json`, loads the recorded base model, and merges the adapter for
inference.

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero_lora.py \
  --task inference \
  --model-config.model-name-or-path ${TRAINING_OUTPUT_DIR}/checkpoint-50000 \
  --model-config.chunk-size 10 \
  --inference-config.output-action-dim 7 \
  --inference-config.image-prompts "Head" "Left wrist" \
  --inference-config.port 7891
```

## 4. Start the Fast Backend

The fast backend uses a TensorRT vision encoder, optimized attention and MLP
kernels, and startup-captured CUDA Graph profiles to reduce latency.

Install the required fast-backend dependency layer:

```bash
pip install -e ".[fast-infer]"
```

`fast-infer` installs `onnx`, `triton==3.6.0`, and `tensorrt`. The fast backend
does not fall back when these pieces are missing: TensorRT is used to prepare
the vision engine, Triton is required by the fast prefix/suffix kernels, and
`flex_attention` must be available in PyTorch.

Add `--inference-config.backend fast` to the matching default backend command.
If `--inference-config.vision-trt-engine-path` is not provided, the default
engine path is `checkpoints/trt_engines/dm05_vision.engine`. For LIBERO:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --model-config.chunk-size 10 \
  --inference-config.backend fast \
  --inference-config.output-action-dim 7 \
  --inference-config.image-prompts "Head" "Left wrist" \
  --inference-config.port 7891
```

If the engine does not exist, the launcher does not start serving immediately.
It first exports ONNX and builds the TensorRT engine, then continues startup and
captures the configured CUDA Graph profiles before the HTTP service becomes
ready. Use a separate engine path for each checkpoint and image layout. An
existing engine is currently reused based on its image count, so changing the
checkpoint requires a new path or `--inference-config.force-rebuild`.

To build an engine manually:

```bash
python -m opendm.infer.build_vision_trt \
  --checkpoint ./checkpoints/DM05-libero \
  --onnx-path checkpoints/trt_engines/dm05_vision.onnx \
  --engine-path checkpoints/trt_engines/dm05_vision.engine \
  --num-images 2
```

`--num-images` defaults to the number of `--inference-config.image-prompts`.
With `--data-config.is-history`, use `len(image_prompts) + 5` (up to 5 history
slots). For three current views plus history, use `--num-images 8`.

### Fast Backend Constraints

- Startup readiness includes TensorRT engine preparation and CUDA Graph profile
  capture; the first launch is expected to take longer than the default backend.
- Requests use batch size 1 and are processed serially by the service.
- `diffusion_steps` is fixed when the service captures its profiles at startup.
- The processed multimodal prefix is limited to 1024 tokens.
- The default prefix buckets are `576 704 768 896 1024`. Defaults too small for
  the configured image count are skipped automatically.
- A custom bucket list must be non-empty, strictly increasing, and no larger
  than 1024.
- More buckets increase service startup time and GPU memory use.
- A request longer than the largest custom bucket uses a slower eager fallback,
  provided it is still within the 1024-token limit.

Override the defaults only when the workload requires different prefix shapes:

```bash
--inference-config.prefix-seq-len-buckets 576 704 768 896 1024
```

All configured profiles are captured before the HTTP service becomes ready.

## 5. Call the HTTP API

The service exposes `POST /v1/infer` with a JSON body. Images are sent as
base64 strings keyed by contiguous 1-based slot names, which keeps the API
consistent across benchmark checkpoints, demo checkpoints, and custom SFT
services.

`POST /process_frame` remains available as a legacy multipart endpoint for
existing clients. New integrations should use `/v1/infer`. The legacy endpoint
will be phased out over time.

A LIBERO request with two images looks like:

```bash
curl -X POST http://127.0.0.1:7891/v1/infer \
  -H 'Content-Type: application/json' \
  --data @- <<'EOF'
{
  "observation": {
    "prompt": "pick up the black bowl and place it on the plate",
    "state": [0, 0, 0, 0, 0, 0, 0, 0],
    "images": {
      "1": "<base64-agentview>",
      "2": "<base64-wrist>"
    },
    "robot_type": "Franka"
  }
}
EOF
```

Request fields:

- `observation.prompt`: task instruction. It defaults to an empty string.
- `observation.state`: required one-dimensional JSON array. Its length and
  ordering must match the checkpoint's normalization statistics.
- `observation.images`: required JSON object of base64-encoded images. Keys must
  be contiguous 1-based strings (`"1"`, `"2"`, …) and map **one-to-one in order**
  to `--inference-config.image-prompts` (e.g. `"1"` → first prompt such as
  `Head`, `"2"` → second such as `Left wrist`).
- `observation.history_images`: optional JSON array of base64 history frames,
  **oldest to newest**, with 5 frames allowed by default. The Default backend can
  accept up to 32 with `--inference-config.max-history-images 32`; also set
  `--trainer-config.model-max-length 2048`. The Fast backend currently remains
  limited to 5. Only valid when the service was started with
  `--data-config.is-history`; omit or use `[]` when there is no history. See
  [`tests/curl_history.sh`](../../tests/curl_history.sh) for a full request.
- `observation.robot_type`: optional robot embodiment used for state/action
  semantics. Benchmark entry points inherit dataset defaults such as `Franka`
  and `Aloha RoboTwin2`. Multi-robot checkpoints select profiles by the exact
  value, for example `Aloha` or `DOS W1`; custom relative-action entries may
  need this field explicitly.
- `observation.control_mode` and `observation.speed`: text-conditioning fields.
  Provide both explicitly when serving the `Dexmal/DM05` base pretrained
  checkpoint. For SFT checkpoints, they are needed only when the training data
  included them. The service defaults `speed` to `"0.5"`.
- `sampling`: optional JSON object. `num_steps` must match the service's fixed
  diffusion steps, and `seed` can be used for deterministic sampling.

A successful response contains the action chunk and end-to-end API latency in
milliseconds:

```json
{
  "actions": [
    [0.012, -0.034, 0.18, 0.0, 0.0, 0.0, -1.0],
    [0.015, -0.031, 0.17, 0.0, 0.0, 0.0, -1.0]
  ],
  "metadata": {
    "latency_ms": 123.4
  }
}
```

For the built-in demo checkpoint you can also run:

```bash
# No history (plain three-image request)
bash tests/curl_demo.sh http://127.0.0.1:7891/v1/infer

# Select the Aloha profile in a multi-robot checkpoint
bash tests/curl_demo.sh http://127.0.0.1:7891/v1/infer Aloha

# With history_images (start the service with --data-config.is-history)
bash tests/curl_history.sh http://127.0.0.1:7891/v1/infer
```

### Legacy `/process_frame` API

The legacy compatibility endpoint accepts `multipart/form-data` with repeated
`image` file fields. Prefer `/v1/infer` JSON for new clients; legacy uses repeated
form fields instead of `"1"` / `"2"` keys.

```bash
curl -X POST http://127.0.0.1:7891/process_frame \
  -F 'text=pick up the black bowl and place it on the plate' \
  -F 'states=[0,0,0,0,0,0,0,0]' \
  -F 'robot_type=Franka' \
  -F image=@/path/to/agentview.jpg \
  -F image=@/path/to/wrist.jpg
```

Legacy request fields:

- `text`: task instruction. It defaults to an empty string.
- `states`: required one-dimensional JSON array. Its length and ordering must
  match the checkpoint's normalization statistics.
- `image`: repeated image file field. The count and order must match
  `image_prompts` (e.g. Head, Left wrist, Right wrist).
- `history_images`: optional repeated history frames (oldest to newest), with 5
  frames allowed by default. The Default backend can be configured for 32,
  while the Fast backend currently remains limited to 5. Only valid when the
  service was started with `--data-config.is-history`.
- `robot_type`: optional robot embodiment used for state/action semantics.
  Benchmark entry points inherit dataset defaults such as `Franka` and `Aloha
  RoboTwin2`. Multi-robot checkpoints select profiles by the exact value, for
  example `Aloha` or `DOS W1`; custom relative-action entries may need this
  field explicitly.
- `control_mode` and `speed`: text-conditioning fields. Provide both explicitly
  when serving the `Dexmal/DM05` base pretrained checkpoint. For SFT
  checkpoints, they are needed only when the training data included them. The
  service defaults `speed` to `"0.5"`.

A successful legacy response returns the historical shape:

```json
{
  "response": [
    [0.012, -0.034, 0.18, 0.0, 0.0, 0.0, -1.0],
    [0.015, -0.031, 0.17, 0.0, 0.0, 0.0, -1.0]
  ],
  "model_latency_ms": 71.884
}
```

## 6. Common Parameters

| Parameter | Meaning |
| --- | --- |
| `--exp` | Playground entry point matching the checkpoint and task. |
| `--model-config.model-name-or-path` | Full model or LoRA checkpoint directory. |
| `--model-config.chunk-size` | Action horizon; must match training and the client. |
| `--trainer-config.model-max-length` | Maximum tokenized multimodal prefix length. |
| `--inference-config.diffusion-steps` | Number of action diffusion steps; default `10`. |
| `--inference-config.output-action-dim` | Returned action dimension; must match normalization statistics. |
| `--inference-config.image-prompts` | Ordered camera labels, one-to-one with `observation.images` keys `"1"`, `"2"`, …. |
| `--inference-config.max-history-images` | Default-backend history request limit. Defaults to `5`; use `32` for 32-frame workloads. The Fast backend remains limited to `5`. |
| `--data-config.is-history` | Required to accept `history_images`; plain `curl_demo.sh` does not need it. |
| `--inference-config.backend` | `default` or `fast`. |
| `--inference-config.vision-trt-engine-path` | Checkpoint-specific TensorRT vision engine path; default `checkpoints/trt_engines/dm05_vision.engine`. |
| `--inference-config.force-rebuild` | Rebuild the vision engine before fast inference. |
| `--inference-config.prefix-seq-len-buckets` | Optional non-empty custom fast-backend buckets. |
| `--inference-config.port` | HTTP service port; default `7891`. |

## 7. Troubleshooting

| Error or symptom | Check |
| --- | --- |
| Missing or mismatched normalization statistics | Use the checkpoint's `norm_stats.json` and the same dataset/action/chunk configuration as training. |
| State or action dimension error | Match `observation.state` and `output_action_dim` to the normalization vectors. |
| Wrong number of uploaded images | Make `observation.images` use the same count and order as `image_prompts`. |
| Fast backend fails during startup with import errors | In the active environment, reinstall `pip install -e ".[fast-infer]"` and verify `import tensorrt`, `import triton`, and `import torch.nn.attention.flex_attention`. |
| TensorRT image-count mismatch | Rebuild with `--num-images` equal to `len(image_prompts)`; with history use `len(image_prompts) + 5`. |
| Results change after switching checkpoints with the same engine | Use a checkpoint-specific engine path or pass `--inference-config.force-rebuild`. |
| Empty, unsorted, or oversized prefix buckets | Pass a non-empty increasing list whose values are at most 1024. |
| Fast prefix exceeds 1024 tokens | Shorten the instruction or reduce `model_max_length`; for 32-frame history use the Default backend with `model_max_length` set to 2048. |
| An occasional default-backend request is slower | A new execution profile runs eagerly once and captures on its second occurrence; later matching requests reuse the Graph. |
| Fast service takes time to become ready | Wait for engine preparation and all configured CUDA Graph profiles to finish at startup. |

## 8. Related Guides

- [DM05 LIBERO Training and Evaluation](dm05_libero.md)
- [DM05 RoboTwin 2.0 Training and Evaluation](dm05_robotwin2.md)
- [DM05 SFT and Validation](dm05_finetuning.md)
- [DM05 LIBERO LoRA Training](dm05_libero_lora_training.md)
