import json
import os

import megfile
import orjson
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


class JsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_dir: str,
        transforms=None,
        dataset_name: str | None = None,
        dataset_meta: dict = {},
    ):
        self.jsonl_dir = jsonl_dir
        self.transforms = transforms
        self.dataset_name = dataset_name
        self.dataset_meta = dataset_meta
        self._build_index()

    def _build_index(self):
        index_cache = self._get_index_cache(self.jsonl_dir)
        file_to_nsamples = index_cache["data"]

        sample_index = []
        file_to_id = {}
        file_id_counter = 0
        for jsonl_file, num_samples in file_to_nsamples.items():
            file_to_id[jsonl_file] = file_id_counter
            for frame_index in range(max(0, num_samples - 1)):
                sample_index.append((file_id_counter, frame_index))
            file_id_counter += 1
        self.sample_index = sample_index
        self.id_to_jsonl = {v: k for k, v in file_to_id.items()}
        self.total_samples = len(self.sample_index)

    def _get_index_cache(self, jsonl_dir: str) -> dict:
        index_cache_file = os.path.join(jsonl_dir, "index_cache.json")
        if megfile.smart_exists(index_cache_file):
            with megfile.smart_open(index_cache_file, "r") as f:
                return json.load(f)
        return build_index_cache(jsonl_dir)

    def __getitem__(self, idx: int) -> dict:
        file_index, frame_index = self.sample_index[idx]
        jsonl_file = self.id_to_jsonl[file_index]
        lines = _read_jsonl_lines(jsonl_file)
        result = orjson.loads(lines[frame_index])
        result["raw_lines"] = lines
        result["meta_data"] = {
            **(self.dataset_meta or {}),
            "frame_index": frame_index,
        }
        if self.transforms is not None:
            result = self.transforms(result)
        return result

    def __len__(self) -> int:
        return self.total_samples


def _read_jsonl_lines(file_path: str) -> list[str]:
    try:
        return open(file_path, "r").readlines()
    except Exception:
        return megfile.smart_open(file_path, "r").readlines()


def _load_jsonl(file_path: str) -> list:
    try:
        f = open(file_path, "r").readlines()
    except Exception:
        f = megfile.smart_open(file_path, "r").readlines()
    return [line for line in f if line.strip()]


def build_index_cache(jsonl_dir: str) -> dict:
    jsonl_files = []
    if megfile.smart_isdir(jsonl_dir):
        logger.info(f"Building index cache for {jsonl_dir} ...")
        jsonl_files = megfile.smart_glob(os.path.join(jsonl_dir, "**", "*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"Dataset not found or empty: {jsonl_dir}")

    data = {}
    total_samples = 0
    for jsonl_file in tqdm(jsonl_files, desc="Building index cache"):
        num_lines = len(_load_jsonl(jsonl_file))
        data[jsonl_file] = num_lines
        total_samples += num_lines
    index_cache = {"data": data}
    index_cache_file = os.path.join(jsonl_dir, "index_cache.json")
    with megfile.smart_open(index_cache_file, "w") as f:
        json.dump(index_cache, f, indent=2)
    logger.info(
        f"Index cache written to {index_cache_file} "
        f"({len(jsonl_files)} files, {total_samples} samples)"
    )
    return index_cache
