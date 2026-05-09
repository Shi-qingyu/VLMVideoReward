import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import torch


DISTILL_ONLY_PREFIXES = ("visual_distill_projector.",)


def _should_strip_key(key: str, prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def filter_state_dict_for_inference(
    state_dict: dict[str, torch.Tensor],
    prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES,
) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if not _should_strip_key(key, prefixes)}


def _load_safetensors():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to sanitize safetensors checkpoints."
        ) from exc
    return load_file, save_file


def _load_safetensors_safe_open():
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to inspect safetensors checkpoints."
        ) from exc
    return safe_open


def _rewrite_torch_weight_file(
    path: Path,
    prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES,
) -> bool:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return False

    cleaned = filter_state_dict_for_inference(state, prefixes)
    if len(cleaned) == len(state):
        return False

    if cleaned:
        torch.save(cleaned, path)
    else:
        path.unlink()
    return True


def _rewrite_safetensors_weight_file(
    path: Path,
    prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES,
) -> bool:
    load_file, save_file = _load_safetensors()
    state = load_file(str(path))
    cleaned = filter_state_dict_for_inference(state, prefixes)
    if len(cleaned) == len(state):
        return False

    if cleaned:
        save_file(cleaned, str(path))
    else:
        path.unlink()
    return True


def _rewrite_index_file(
    path: Path,
    prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES,
) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        return False

    cleaned_weight_map = {
        key: value
        for key, value in weight_map.items()
        if not _should_strip_key(key, prefixes)
    }
    if len(cleaned_weight_map) == len(weight_map):
        return False

    index_data["weight_map"] = cleaned_weight_map
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    return True


def _bin_file_has_strippable_keys(
    path: Path,
    prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES,
) -> bool:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return False
    return any(_should_strip_key(key, prefixes) for key in state.keys())


def _safetensors_file_has_strippable_keys(
    path: Path,
    prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES,
) -> bool:
    safe_open = _load_safetensors_safe_open()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return any(_should_strip_key(key, prefixes) for key in f.keys())


def model_dir_needs_inference_sanitization(
    model_dir: str | Path,
    prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES,
) -> bool:
    model_dir = Path(model_dir)

    for path in sorted(model_dir.glob("*.index.json")):
        with open(path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        if any(_should_strip_key(key, prefixes) for key in weight_map.keys()):
            return True

    for path in sorted(model_dir.glob("*.safetensors")):
        if _safetensors_file_has_strippable_keys(path, prefixes):
            return True

    for path in sorted(model_dir.glob("*.bin")):
        if _bin_file_has_strippable_keys(path, prefixes):
            return True

    return False


def strip_distill_only_weights_in_dir(
    model_dir: str | Path,
    prefixes: Iterable[str] = DISTILL_ONLY_PREFIXES,
) -> bool:
    model_dir = Path(model_dir)
    changed = False

    for path in sorted(model_dir.glob("*.safetensors")):
        changed = _rewrite_safetensors_weight_file(path, prefixes) or changed
    for path in sorted(model_dir.glob("*.bin")):
        changed = _rewrite_torch_weight_file(path, prefixes) or changed
    for path in sorted(model_dir.glob("*.index.json")):
        changed = _rewrite_index_file(path, prefixes) or changed

    return changed


def _copy_inference_artifacts(src_dir: Path, dst_dir: Path) -> None:
    for item in src_dir.iterdir():
        if item.name.startswith("checkpoint-"):
            continue
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)


def prepare_inference_model_dir(model_dir: str | Path) -> str:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return str(model_dir)
    if not model_dir_needs_inference_sanitization(model_dir):
        return str(model_dir)

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{model_dir.name}-inference-"))
    _copy_inference_artifacts(model_dir, temp_dir)
    changed = strip_distill_only_weights_in_dir(temp_dir)
    if not changed:
        shutil.rmtree(temp_dir)
        return str(model_dir)
    return str(temp_dir)
