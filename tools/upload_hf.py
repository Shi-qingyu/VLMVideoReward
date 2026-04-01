import os
import argparse
from huggingface_hub import HfApi


HF_TOKEN = 
REPO_ID = "QingyuShi/videoreward"
REPO_TYPE = "dataset"  # change to "model" if your repo is a model repo


def upload_file_to_hf(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_name = os.path.basename(file_path)
    path_in_repo = file_name

    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    print(f"Uploaded successfully: {file_path}")
    print(f"Repo: {REPO_ID}")
    print(f"Path in repo: ./{file_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to the local file")
    args = parser.parse_args()

    upload_file_to_hf(args.file_path)


if __name__ == "__main__":
    main()