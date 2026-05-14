import argparse
import os


ENTRYPOINTS = {
    "qwen3vl": "inference_qwen.py",
    "qwen2.5vl": "inference_qwen.py",
    "qwen2vl": "inference_qwen.py",
    "internvl": "inference_internvl.py",
    "molmo2": "inference_molmo2.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compatibility shim. Use a model-specific inference file instead.",
    )
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument(
        "--model_type",
        default=os.environ.get("MODEL_TYPE", "auto"),
        choices=[
            "auto",
            "qwen3vl",
            "qwen2.5vl",
            "qwen2vl",
            "internvl",
            "gemma4",
            "minicpmv",
            "molmo2",
        ],
    )
    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    if not args.model_path:
        raise SystemExit(
            "Use a model-specific entrypoint, e.g. "
            "`python inference_qwen.py --model_path ...` or "
            "`python inference_internvl.py --model_path ...`."
        )

    if args.model_type == "auto":
        from inference_common import infer_model_type
        from src.train.checkpoint_utils import prepare_inference_model_dir

        inference_model_path = prepare_inference_model_dir(args.model_path)
        model_type = infer_model_type(inference_model_path)
    else:
        model_type = args.model_type
    entrypoint = ENTRYPOINTS.get(model_type)
    if entrypoint is None:
        raise SystemExit(
            f"No standalone inference entrypoint is configured for {model_type} yet."
        )

    raise SystemExit(
        f"{model_type} now uses `{entrypoint}`. Run:\n"
        f"python {entrypoint} --model_path {args.model_path}"
    )


if __name__ == "__main__":
    main()
