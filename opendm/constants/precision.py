"""Mixed-precision policy constants for OpenDM."""

import torch

BF16_MIXED_PRECISION_POLICY = "bf16_mixed"
FP32_MIXED_PRECISION_POLICY = "fp32_mixed"
MODEL_DTYPE = torch.float32
COMPUTE_DTYPE = torch.bfloat16
FSDP_MIXED_PRECISION = "fp32"
TF32_ENABLED = False
