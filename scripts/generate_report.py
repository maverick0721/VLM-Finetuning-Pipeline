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

    return path


def generate_markdown(q, u, plots):

    md = []

    md.append("# Vision-Language Model Fine-Tuning Experiment\n")

    md.append("## Experiment Setup\n")

    md.append("- Dataset size: 100 images\n")
    md.append("- Model: LLaVA 1.5 7B\n")
    md.append("- Training methods compared: QLoRA vs Unsloth\n")

    md.append("\n## Benchmark Results\n")

    md.append("| Metric | QLoRA | Unsloth |\n")
    md.append("|------|------|------|\n")

    for k in q:
        md.append(f"| {k} | {q[k]} | {u[k]} |\n")

    md.append("\n## Training Performance Charts\n")

    for plot in plots:
        md.append(f"![{plot}]({plot})\n")

    md.append("\n## Observations\n")

    md.append(
        "Unsloth generally provides faster training throughput "
        "while maintaining comparable performance.\n"
    )

    path = f"{OUTPUT_DIR}/experiment_report.md"

    with open(path, "w") as f:
        f.writelines(md)

    print("Report generated:", path)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    q, u = load_results()

    plots = []

    for metric in q:

        plot = plot_metric(metric, q[metric], u[metric])

        plots.append(plot)

    generate_markdown(q, u, plots)


if __name__ == "__main__":
    main()