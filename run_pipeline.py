import os
import json
from dotenv import load_dotenv


def run(cmd):
    print("\n========================================")
    print(f"Running: {cmd}")
    print("========================================\n")

    result = os.system(cmd)

    if result != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def load_environment():

    print("Loading environment variables...")

    load_dotenv()

    wandb_key = os.getenv("WANDB_API_KEY")

    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        print("WANDB API key loaded")
    else:
        print("Warning: WANDB_API_KEY not found")


def compare_results():

    qlora_file = "models/qlora/results.json"
    unsloth_file = "models/unsloth/results.json"

    if not os.path.exists(qlora_file) or not os.path.exists(unsloth_file):
        print("No evaluation results found for comparison")
        return

    with open(qlora_file) as f:
        qlora = json.load(f)

    with open(unsloth_file) as f:
        unsloth = json.load(f)

    print("\n===============================")
    print(" FINAL COMPARISON REPORT")
    print("===============================\n")

    for metric in qlora:

        q_val = qlora[metric]
        u_val = unsloth.get(metric)

        print(f"{metric}")
        print(f"  QLoRA   : {q_val}")
        print(f"  Unsloth : {u_val}")
        print()

    print("Comparison complete.\n")


def main():

    load_environment()

    # Step 1: Download Dataset
    run("python scripts/download_dataset.py")


    # Step 2: Prepare Dataset
    run("python scripts/prepare_dataset.py")

    
    # Step 3: Train QLoRA
    print("\nStarting QLoRA training...\n")

    run("python scripts/train_qlora.py")

    run("python scripts/evaluate.py --model models/qlora")

   
    # Step 4: Train Unsloth
    print("\nStarting Unsloth training...\n")

    run("python scripts/train_unsloth.py")

    run("python scripts/evaluate.py --model models/unsloth")

    
    # Step 5: Compare Results
    compare_results()

    print("\nPipeline completed successfully.\n")


if __name__ == "__main__":
    main()