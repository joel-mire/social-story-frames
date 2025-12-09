"""
Upload SSF LoRA-finetuned models to HuggingFace Hub (private).

Usage:
    # First login to HuggingFace
    huggingface-cli login

    # Then upload models
    cd scripts
    python upload_models_to_hf.py --hf_username YOUR_HF_USERNAME

Requirements:
    - huggingface_hub (pip install huggingface_hub)
    - You must be logged in: huggingface-cli login
"""

import argparse
import yaml
from huggingface_hub import HfApi, create_repo
from ssf.Configs import Config
from ssf.Constants import *

def load_config(path: str) -> Config:
    """Load config as Pydantic object."""
    with open(path, 'r') as f:
        return Config(**yaml.safe_load(f))


def upload_model(local_path: str, repo_id: str, model_name: str, dry_run: bool = False):
    """Upload model to HuggingFace Hub."""
    import os

    print(f"\n{'='*80}")
    print(f"{'[DRY RUN] ' if dry_run else ''}Uploading {model_name}")
    print(f"{'='*80}")
    print(f"Local:  {local_path}")
    print(f"Repo:   {repo_id}")

    if not os.path.exists(local_path):
        print(f"[ERROR] Path does not exist: {local_path}")
        return False

    # Show key files
    print(f"\nKey files:")
    for file in ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json']:
        file_path = os.path.join(local_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {file} (missing)")

    if dry_run:
        print(f"\n[DRY RUN] Would upload to {repo_id}")
        return True

    try:
        api = HfApi()

        # Create private repo
        print(f"\nCreating private repository...")
        create_repo(repo_id=repo_id, private=True, repo_type="model", exist_ok=True)

        # Upload
        print(f"Uploading...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {model_name} checkpoint",
            ignore_patterns=["checkpoint-*"]
        )

        print(f"[SUCCESS] https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload SSF models to HuggingFace")
    parser.add_argument('--hf_username', type=str, required=True, help='HuggingFace username')
    parser.add_argument('--config', type=str, default=REPLICATION_CONFIG_PATH, help='Config path')
    parser.add_argument('--dry-run', action='store_true', help='Preview without uploading')
    args = parser.parse_args()

    print("SSF Model Upload to HuggingFace Hub")
    print("="*80)

    # Load config
    print(f"\nLoading: {args.config}")
    config = load_config(args.config)
    print(f"Config ID: {config.id}")

    # Build paths from config
    gen_path = f"{config.dirs.data.ssf_gen_ft}/ft-models/{config.ft.ssf_gen_ft_script_base_name}-train-prompt_default"
    class_path = f"{config.dirs.data.ssf_class_ft}/ft-models/{config.ft.ssf_class_ft_script_base_name}-train-zero-shot-gen0"

    print(f"\nPaths:")
    print(f"  Generator:  {gen_path}")
    print(f"  Classifier: {class_path}")

    # Upload both models
    results = [
        upload_model(gen_path, f"{args.hf_username}/llama3.1-8b-it-ssf-generator",
                    "SSF-Generator", args.dry_run),
        upload_model(class_path, f"{args.hf_username}/llama3.1-8b-it-ssf-classifier",
                    "SSF-Classifier", args.dry_run)
    ]

    # Summary
    print(f"\n{'='*80}")
    print(f"Result: {sum(results)}/2 models uploaded")
    if args.dry_run:
        print("[DRY RUN] Remove --dry-run to upload")
    print("="*80)


if __name__ == "__main__":
    main()
