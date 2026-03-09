import os
import json
from dotenv import load_dotenv


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

    load_env()

   
    # DATASETUP
    run("python -m scripts.download_dataset")
    run("python -m scripts.prepare_dataset")

   
    # QLoRA
    print("\nRunning QLoRA experiment...\n")

    run("python -m scripts.train_qlora")
    run("python -m scripts.evaluate --model models/qlora")

    
    # Unsloth
    print("\nRunning Unsloth experiment...\n")

    run("python -m scripts.train_unsloth")
    run("python -m scripts.evaluate --model models/unsloth")

  
    # Benchmark comparison
    compare()

   
    # Generate plots
    run("python -m scripts.generate_report")

  
    # Launch demo
    print("\nLaunching interactive demo...\n")

    run("python -m scripts.demo.py")

    print("\nPipeline finished successfully\n")


if __name__ == "__main__":
    main()