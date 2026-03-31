import json
import os

import matplotlib.pyplot as plt


QLORA_FILE = "models/qlora/benchmark.json"
UNSLOTH_FILE = "models/unsloth/benchmark.json"
OUTPUT_DIR = "reports"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_results():
    q = load_json(QLORA_FILE) if os.path.exists(QLORA_FILE) else None
    u = load_json(UNSLOTH_FILE) if os.path.exists(UNSLOTH_FILE) else None

    if q is None and u is None:
        raise RuntimeError("No benchmark files found. Run training first.")

    return q, u


def plot_metric(metric, q_value, u_value):
    methods = []
    values = []

    if q_value is not None:
        methods.append("QLoRA")
        values.append(q_value)

    if u_value is not None:
        methods.append("Unsloth")
        values.append(u_value)

    plt.figure()
    plt.bar(methods, values)
    plt.title(metric)
    plt.ylabel(metric)

    path = f"{OUTPUT_DIR}/{metric}.png"
    plt.savefig(path)
    plt.close()

    return path


def generate_markdown(q, u, plots):
    q = q or {}
    u = u or {}

    metrics = sorted(set(q.keys()) | set(u.keys()))

    md = []
    md.append("# Vision-Language Model Fine-Tuning Experiment\n")
    md.append("## Experiment Setup\n")
    md.append("- Dataset size: 100 images\n")
    md.append("- Model: LLaVA 1.5 7B\n")
    md.append("- Training methods compared: QLoRA vs Unsloth\n")

    md.append("\n## Benchmark Results\n")
    md.append("| Metric | QLoRA | Unsloth |\n")
    md.append("|------|------|------|\n")

    for k in metrics:
        md.append(f"| {k} | {q.get(k, 'N/A')} | {u.get(k, 'N/A')} |\n")

    md.append("\n## Training Performance Charts\n")
    for plot in plots:
        md.append(f"![{plot}]({plot})\n")

    md.append("\n## Observations\n")
    md.append("Throughput and memory trends depend on hardware and model backend details.\n")

    path = f"{OUTPUT_DIR}/experiment_report.md"
    with open(path, "w") as f:
        f.writelines(md)

    print("Report generated:", path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    q, u = load_results()

    plots = []
    metrics = sorted(set((q or {}).keys()) | set((u or {}).keys()))

    for metric in metrics:
        plot = plot_metric(metric, (q or {}).get(metric), (u or {}).get(metric))
        plots.append(plot)

    generate_markdown(q, u, plots)


if __name__ == "__main__":
    main()
