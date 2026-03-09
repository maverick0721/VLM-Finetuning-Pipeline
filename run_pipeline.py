import os
import json
from dotenv import load_dotenv


RAW_DATA_DIR = "data/raw/images"
PROCESSED_DATA = "data/processed/train.json"


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


def dataset_exists():

    return os.path.exists(RAW_DATA_DIR) and len(os.listdir(RAW_DATA_DIR)) > 0


def processed_exists():

    return os.path.exists(PROCESSED_DATA)


def compare():

    q_file = "models/qlora/benchmark.json"
    u_file = "models/unsloth/benchmark.json"

    if not os.path.exists(q_file) or not os.path.exists(u_file):
        return

    with open(q_file) as f:
        q = json.load(f)

    with open(u_file) as f:
        u = json.load(f)

    print("\n========== BENCHMARK ==========\n")

    for k in q:

        print(k)
        print("QLoRA:", q[k])
        print("Unsloth:", u.get(k))
        print()


def main():

    load_env()

   
    # DATA DOWNLOAD
    if not dataset_exists():
        run("python -m scripts.download_dataset")
    else:
        print("Dataset already exists")

    
    # DATA PREPARATION
    if not processed_exists():
        run("python -m scripts.prepare_dataset")
    else:
        print("Processed dataset already exists")

  
    # TRAINING & EVALUATION
    run("python -m scripts.train_qlora")
    run("python -m scripts.evaluate --model models/qlora")

    run("python -m scripts.train_unsloth")
    run("python -m scripts.evaluate --model models/unsloth")

  
    # BENCHMARK
    compare()

    run("python -m scripts.generate_report")

    
    # DEMO
    run("python -m scripts.demo")


if __name__ == "__main__":
    main()