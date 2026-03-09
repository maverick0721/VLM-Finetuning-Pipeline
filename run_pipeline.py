import os
import json
import subprocess
from dotenv import load_dotenv


RAW_DATA_DIR = "data/raw/images"
PROCESSED_DATA = "data/processed/train.json"

QLORA_MODEL_DIR = "models/qlora"
UNSLOTH_MODEL_DIR = "models/unsloth"


def run(cmd):

    print("\n=======================================")
    print("Running:", cmd)
    print("=======================================\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


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


def model_exists(path):

    return os.path.exists(path) and len(os.listdir(path)) > 0


def compare():

    q_file = "models/qlora/benchmark.json"
    u_file = "models/unsloth/benchmark.json"

    if not os.path.exists(q_file) or not os.path.exists(u_file):
        print("Benchmark files not found. Skipping comparison.")
        return

    with open(q_file) as f:
        q = json.load(f)

    with open(u_file) as f:
        u = json.load(f)

    print("\n========== BENCHMARK ==========\n")

    for k in q:

        print(k)
        print("QLoRA  :", q[k])
        print("Unsloth:", u.get(k))
        print()


def main():

    load_env()

    # DATA DOWNLOAD
    print("\nSTEP 1: DATA DOWNLOAD")

    if not dataset_exists():
        run("python -m scripts.download_dataset")
    else:
        print("Dataset already exists")

    # DATA PREPARATION
    print("\nSTEP 2: DATA PREPARATION")

    if not processed_exists():
        run("python -m scripts.prepare_dataset")
    else:
        print("Processed dataset already exists")

    # QLORA TRAINING & EVALUATION
    print("\nSTEP 3: QLoRA TRAINING")

    if not model_exists(QLORA_MODEL_DIR):
        run("python -m scripts.train_qlora")
        run("python -m scripts.evaluate --model models/qlora")
    else:
        print("QLoRA model already trained")

    # UNSLOTH TRAINING & EVALUATION
    print("\nSTEP 4: UNSLOTH TRAINING")

    if not model_exists(UNSLOTH_MODEL_DIR):
        run("python -m scripts.train_unsloth")
        run("python -m scripts.evaluate --model models/unsloth")
    else:
        print("Unsloth model already trained")

    # BENCHMARK COMPARISON
    print("\nSTEP 5: BENCHMARK COMPARISON")

    compare()

    # REPORT GENERATION
    print("\nSTEP 6: GENERATE REPORT")

    run("python -m scripts.generate_report")

    # DIAGRAM GENERATION
    print("\nSTEP 7: GENERATE DIAGRAMS")

    run("python -m scripts.generate_diagrams")

    # REPORT+DIAGRAM GENERATION
    print("\nSTEP 8: GENERATE FULL REPORT PDF")

    run("python -m scripts.export_report_pdf")

    # DEMO LAUNCH
    print("\nSTEP 9: LAUNCH DEMO")

    run("python -m scripts.demo")


if __name__ == "__main__":
    main()