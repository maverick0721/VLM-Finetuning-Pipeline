import json
import os
import matplotlib.pyplot as plt


QLORA_FILE = "models/qlora/benchmark.json"
UNSLOTH_FILE = "models/unsloth/benchmark.json"

OUTPUT_DIR = "reports"


def load_results():

    with open(QLORA_FILE) as f:
        q = json.load(f)

    with open(UNSLOTH_FILE) as f:
        u = json.load(f)

    return q, u


def plot_metric(metric, q_value, u_value):

    methods = ["QLoRA", "Unsloth"]
    values = [q_value, u_value]

    plt.figure()

    plt.bar(methods, values)

    plt.title(metric)
    plt.ylabel(metric)

    path = f"{OUTPUT_DIR}/{metric}.png"

    plt.savefig(path)

    print("Saved:", path)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    q, u = load_results()

    for metric in q:

        plot_metric(metric, q[metric], u.get(metric))

    print("Reports generated")


if __name__ == "__main__":
    main()