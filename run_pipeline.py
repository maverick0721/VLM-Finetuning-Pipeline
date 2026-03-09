import os
import argparse
from dotenv import load_dotenv


def run_command(command):
    print("\n--------------------------------------------------")
    print(f"Running: {command}")
    print("--------------------------------------------------\n")

    exit_code = os.system(command)

    if exit_code != 0:
        raise RuntimeError(f"Command failed: {command}")


def setup_env():
    print("Loading environment variables...")

    load_dotenv()

    wandb_key = os.getenv("WANDB_API_KEY")

    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        print("WANDB_API_KEY loaded from .env")
    else:
        print("Warning: WANDB_API_KEY not found in .env")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        default="qlora",
        choices=["qlora", "unsloth"],
        help="Training method"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download"
    )

    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip dataset preprocessing"
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training"
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation"
    )

    args = parser.parse_args()

    setup_env()

   
    # Step 1: Download Dataset
    if not args.skip_download:
        run_command("python scripts/download_dataset.py")


    # Step 2: Prepare Dataset
    if not args.skip_preprocess:
        run_command("python scripts/prepare_dataset.py")


    # Step 3: Training
    if not args.skip_train:

        if args.method == "qlora":
            run_command("python scripts/train_qlora.py")

        elif args.method == "unsloth":
            run_command("python scripts/train_unsloth.py")


    # Step 4: Evaluation
    if not args.skip_eval:
        run_command("python scripts/evaluate.py")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()