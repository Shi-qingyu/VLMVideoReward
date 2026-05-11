import os
import json
import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText


def find_weight_files(model_path: str):
    model_path = Path(model_path)

    # 优先 safetensors
    safetensors_index = model_path / "model.safetensors.index.json"
    if safetensors_index.exists():
        with open(safetensors_index, "r", encoding="utf-8") as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        return [model_path / f for f in files], "safetensors"

    safetensors_file = model_path / "model.safetensors"
    if safetensors_file.exists():
        return [safetensors_file], "safetensors"

    # 兼容 pytorch bin
    bin_index = model_path / "pytorch_model.bin.index.json"
    if bin_index.exists():
        with open(bin_index, "r", encoding="utf-8") as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        return [model_path / f for f in files], "bin"

    bin_file = model_path / "pytorch_model.bin"
    if bin_file.exists():
        return [bin_file], "bin"

    raise FileNotFoundError(
        f"No model weight file found in {model_path}. "
        "Expected model.safetensors / model.safetensors.index.json / "
        "pytorch_model.bin / pytorch_model.bin.index.json"
    )


def load_state_file(path: Path, file_type: str):
    if file_type == "safetensors":
        return safe_load_file(str(path), device="cpu")

    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "model" in obj:
            return obj["model"]
    return obj


def copy_non_weight_files(src_dir: str, dst_dir: str):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    skip_suffixes = {
        ".bin",
        ".safetensors",
    }

    skip_names = {
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }

    for item in src_dir.iterdir():
        if item.name in skip_names:
            continue
        if item.suffix in skip_suffixes:
            continue
        if item.is_dir():
            if item.name in ["checkpoint", "global_step"]:
                continue
            dst_item = dst_dir / item.name
            if dst_item.exists():
                shutil.rmtree(dst_item)
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_dir / item.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Original checkpoint dir")
    parser.add_argument("--dst", type=str, required=True, help="Clean checkpoint output dir")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--save_safetensors", action="store_true", default=True)
    args = parser.parse_args()

    src = args.src
    dst = args.dst

    os.makedirs(dst, exist_ok=True)

    print(f"Loading config from: {src}")
    config = AutoConfig.from_pretrained(
        src,
        trust_remote_code=args.trust_remote_code,
    )

    print("Building model from config...")
    model = AutoModelForImageTextToText.from_config(
        config,
        trust_remote_code=args.trust_remote_code,
    )

    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    weight_files, file_type = find_weight_files(src)
    print(f"Found {len(weight_files)} weight file(s), type={file_type}")

    filtered_state = {}
    unexpected_keys = []
    mismatched_keys = []

    for wf in weight_files:
        print(f"Reading: {wf.name}")
        state = load_state_file(wf, file_type)

        for k, v in state.items():
            # 兼容一些训练框架保存的 module. 前缀
            original_k = k
            if k.startswith("module."):
                k = k[len("module."):]

            if k not in model_keys:
                unexpected_keys.append(original_k)
                continue

            if tuple(v.shape) != tuple(model_state[k].shape):
                mismatched_keys.append(
                    {
                        "key": original_k,
                        "ckpt_shape": tuple(v.shape),
                        "model_shape": tuple(model_state[k].shape),
                    }
                )
                continue

            filtered_state[k] = v

    missing_keys = [k for k in model_keys if k not in filtered_state]

    print("\n========== Clean Summary ==========")
    print(f"Kept keys:        {len(filtered_state)}")
    print(f"Unexpected keys:  {len(unexpected_keys)}")
    print(f"Mismatched keys:  {len(mismatched_keys)}")
    print(f"Missing keys:     {len(missing_keys)}")

    if unexpected_keys:
        print("\nExample unexpected keys:")
        for k in unexpected_keys[:20]:
            print(f"  {k}")

    if mismatched_keys:
        print("\nExample mismatched keys:")
        for item in mismatched_keys[:20]:
            print(
                f"  {item['key']}: "
                f"ckpt={item['ckpt_shape']} model={item['model_shape']}"
            )

    if missing_keys:
        print("\nExample missing keys:")
        for k in missing_keys[:20]:
            print(f"  {k}")

    print("\nLoading filtered state dict with strict=False...")
    msg = model.load_state_dict(filtered_state, strict=False)
    print(msg)

    print(f"\nSaving clean model to: {dst}")
    model.save_pretrained(
        dst,
        safe_serialization=args.save_safetensors,
    )

    print("Copying processor/tokenizer/config related files...")
    try:
        processor = AutoProcessor.from_pretrained(
            src,
            trust_remote_code=args.trust_remote_code,
        )
        processor.save_pretrained(dst)
    except Exception as e:
        print(f"Warning: failed to save processor via AutoProcessor: {e}")
        print("Fallback: copying non-weight files.")
        copy_non_weight_files(src, dst)

    print("\nDone.")
    print(f"Clean checkpoint saved at: {dst}")


if __name__ == "__main__":
    main()