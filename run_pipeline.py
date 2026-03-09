import os
import json
import argparse
from dotenv import load_dotenv


RAW_DATA_DIR = "data/raw/images"
PROCESSED_DATA = "data/processed/train.json"

QLORA_MODEL = "models/qlora"
UNSLOTH_MODEL = "models/unsloth"


def run(cmd):

    print("\n=======================================")
    print("Running:", cmd)
    print("=======================================\n")

    if os.system(cmd) != 0:
        raise RuntimeError("Command failed")


def load_env():

    load_dotenv()

    key = os.getenv("WANDB_API_KEY")

    if key:
        os.environ["WANDB_API_KEY"] = key
        print("WANDB key loaded")
    else:
        print("Warning: WANDB_API_KEY not found")


def dataset_exists():
    return os.path.exists(RAW_DATA_DIR) and len(os.listdir(RAW_DATA_DIR)) > 0


def processed_exists():
    return os.path.exists(PROCESSED_DATA)


def qlora_exists():
    return os.path.exists(QLORA_MODEL)


def unsloth_exists():
    return os.path.exists(UNSLOTH_MODEL)


def compare():

    q_file = "models/qlora/benchmark.json"
    u_file = "models/unsloth/benchmark.json"

    if not os.path.exists(q_file) or not os.path.exists(u_file):
        print("Benchmark files missing")
        return

    with open(q_file) as f:
        q = json.load(f)

    with open(u_file) as f:
        u = json.load(f)

    print("\n============================")
    print("BENCHMARK COMPARISON")
    print("============================\n")

    for k in q:
        print(k)
        print("  QLoRA   :", q[k])
        print("  Unsloth :", u.get(k))
        print()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--skip-demo", action="store_true")

    args = parser.parse_args()

    load_env()


    # DATA DOWNLOAD
    if not dataset_exists():
        run("python -m scripts.download_dataset")
    else:
        print("Dataset already exists, skipping download")

   
    # DATA PREPARATION
    if not processed_exists():
        run("python -m scripts.prepare_dataset")
    else:
        print("Processed dataset already exists")

   
    # QLoRA TRAINING
    if not qlora_exists():
        print("\nTraining QLoRA...\n")
        run("python -m scripts.train_qlora")
        run("python -m scripts.evaluate --model models/qlora")
    else:
        print("QLoRA model already trained")

    
    # UNSLOTH TRAINING
    if not unsloth_exists():
        print("\nTraining Unsloth...\n")
        run("python -m scripts.train_unsloth")
        run("python -m scripts.evaluate --model models/unsloth")
    else:
        print("Unsloth model already trained")

    
    # BENCHMARK COMPARISON
    compare()

  
    # REPORT GENERATION
    run("python -m scripts.generate_report")

   
    # DEMO
    if not args.skip_demo:
        print("\nLaunching demo...\n")
        run("python -m scripts.demo")
    else:
        print("Demo skipped")


if __name__ == "__main__":
    main()