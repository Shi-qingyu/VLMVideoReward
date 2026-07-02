import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

EVALUATION_DIR = Path(__file__).resolve().parent


ENTRYPOINTS = {
    "qwen3vl": EVALUATION_DIR / "evaluation_qwen.py",
    "qwen2.5vl": EVALUATION_DIR / "evaluation_qwen.py",
    "qwen2vl": EVALUATION_DIR / "evaluation_qwen.py",
    "internvl": EVALUATION_DIR / "evaluation_internvl.py",
    "gemma4": EVALUATION_DIR / "evaluation_gemma4.py",
    "minicpmv": EVALUATION_DIR / "evaluation_minicpmv.py",
    "molmo2": EVALUATION_DIR / "evaluation_molmo2.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compatibility shim. Use a model-specific evaluation file instead.",
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
            "`python src/evaluation/evaluation_qwen.py --model_path ...` or "
            "`python src/evaluation/evaluation_internvl.py --model_path ...`."
        )

    if args.model_type == "auto":
        from inference_common import infer_model_type

        model_type = infer_model_type(args.model_path)
    else:
        model_type = args.model_type

    entrypoint = ENTRYPOINTS.get(model_type)
    if entrypoint is None:
        raise SystemExit(
            f"No standalone evaluation entrypoint is configured for {model_type} yet."
        )

    raise SystemExit(
        f"{model_type} now uses `{entrypoint}`. Run:\n"
        f"python {entrypoint} --model_path {args.model_path}"
    )


if __name__ == "__main__":
    main()
