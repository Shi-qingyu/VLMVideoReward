import os

from src.train.train_qwen_sft import train


if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
