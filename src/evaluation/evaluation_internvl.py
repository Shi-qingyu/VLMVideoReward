import argparse
import multiprocessing as mp

try:
    from .evaluation_common import add_common_eval_args, run_eval
except ImportError:
    from evaluation_common import add_common_eval_args, run_eval


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate InternVL checkpoints.")
    parser.add_argument("--model_type", default="auto", choices=["auto", "internvl"])
    add_common_eval_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_eval(get_args(), allowed_model_types={"internvl"}, default_backend="hf")
