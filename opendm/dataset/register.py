import importlib
import os
import sys

CONVERSATION_DATA = {}


def register_dataset(dataset, prefix=""):
    if prefix:
        dataset = {f"{prefix}_{k}": v for k, v in dataset.items()}
    CONVERSATION_DATA.update(dataset)


_DEFAULT_REGISTRY_DIR = os.path.dirname(__file__)


def _import_registry_modules(directory: str) -> None:
    if not os.path.isdir(directory):
        return

    if directory not in sys.path:
        sys.path.insert(0, directory)

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        if filename in ("__init__.py", "register.py"):
            continue
        importlib.import_module(filename[:-3])


_import_registry_modules(_DEFAULT_REGISTRY_DIR)

_extra_registry_path = os.getenv("OPENDM_DATA_PATH")
if _extra_registry_path:
    _import_registry_modules(_extra_registry_path)
