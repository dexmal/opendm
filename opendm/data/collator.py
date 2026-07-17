"""Batch collation for training and norm-stat computation."""

from collections.abc import Sequence

import numpy as np
import torch
from loguru import logger


class TrainingCollator:
    def __init__(
        self,
        pad_token_id: int,
        max_length: int | None = 1024,
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        batch = {}

        input_ids_list = [inst["input_ids"] for inst in instances]
        attention_mask_list = [inst["attention_mask"] for inst in instances]
        token_type_ids_list = [inst["token_type_ids"] for inst in instances]

        padded_input_ids = []
        padded_attention_mask = []
        padded_token_type_ids = []

        max_len = self.max_length
        for input_ids, attention_mask, token_type_ids in zip(
            input_ids_list, attention_mask_list, token_type_ids_list, strict=True
        ):
            seq_len = input_ids.shape[1]
            if seq_len > max_len:
                logger.warning(
                    "Input sequence length {} exceeds max_length {}; truncating.",
                    seq_len,
                    max_len,
                )
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]
                token_type_ids = token_type_ids[:, :max_len]
                seq_len = max_len

            pad_len = max_len - seq_len
            if pad_len > 0:
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (
                                1,
                                pad_len,
                            ),
                            self.pad_token_id,
                            dtype=input_ids.dtype,
                        ),
                    ],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.zeros((1, pad_len), dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                token_type_ids = torch.cat(
                    [
                        token_type_ids,
                        torch.zeros((1, pad_len), dtype=token_type_ids.dtype),
                    ],
                    dim=1,
                )
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_token_type_ids.append(token_type_ids)

        batch["input_ids"] = torch.cat(padded_input_ids, dim=0)
        batch["attention_mask"] = torch.cat(padded_attention_mask, dim=0)
        batch["token_type_ids"] = torch.cat(padded_token_type_ids, dim=0)

        pixel_values_list = [inst["pixel_values"] for inst in instances]
        action_list = [inst["action"] for inst in instances]
        action_mask_list = [inst["action_mask"] for inst in instances]

        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        batch["action"] = torch.cat(action_list, dim=0)
        batch["action_mask"] = torch.cat(action_mask_list, dim=0)

        return batch


class NormStatsCollator:
    def __call__(self, instances: Sequence[dict]) -> dict[str, np.ndarray]:
        batch = {}

        state_list = [inst["state"] for inst in instances]
        action_list = [inst["action"] for inst in instances]

        batch["state"] = np.stack(state_list, axis=0)
        batch["action"] = np.concatenate(action_list, axis=0)

        return batch
