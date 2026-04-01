import json
import os

import matplotlib.pyplot as plt


QLORA_DIR = "models/qlora"
UNSLOTH_DIR = "models/unsloth"
DATA_PATH = "data/processed/train.json"
OUTPUT_DIR = "reports"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_model_results(model_dir):
    results = {}

    benchmark_path = os.path.join(model_dir, "benchmark.json")
    evaluation_path = os.path.join(model_dir, "evaluation.json")

    if os.path.exists(benchmark_path):
        results.update(load_json(benchmark_path))
    if os.path.exists(evaluation_path):
        results.update(load_json(evaluation_path))

    return results


def load_results():
    q = load_model_results(QLORA_DIR)
    u = load_model_results(UNSLOTH_DIR)

    if not q and not u:
        raise RuntimeError("No benchmark or evaluation files found. Run training first.")

    return q, u


def load_dataset_size():
    if not os.path.exists(DATA_PATH):
        return None

    with open(DATA_PATH) as f:
        data = json.load(f)

    return len(data)


def numeric_metrics(q, u):
    metrics = []
    for key in sorted(set(q.keys()) | set(u.keys())):
        q_value = q.get(key)
        u_value = u.get(key)
        if isinstance(q_value, (int, float)) or isinstance(u_value, (int, float)):
            metrics.append(key)
    return metrics


def plot_metric(metric, q_value, u_value):
    methods = []
    values = []

    if isinstance(q_value, (int, float)):
        methods.append("QLoRA")
        values.append(q_value)

    if isinstance(u_value, (int, float)):
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


def generate_markdown(q, u, plots, dataset_size):
    metrics = numeric_metrics(q, u)

    md = []
    md.append("# Vision-Language Model Fine-Tuning Experiment\n")
    md.append("## Experiment Setup\n")
    md.append(f"- Current cleaned dataset size: {dataset_size if dataset_size is not None else 'unknown'} samples\n")
    md.append("- Model: LLaVA 1.5 7B\n")
    md.append("- Training methods compared: QLoRA vs Unsloth\n")

    md.append("\n## Aggregate Results\n")
    md.append("| Metric | QLoRA | Unsloth |\n")
    md.append("|------|------|------|\n")

    for key in metrics:
        md.append(f"| {key} | {q.get(key, 'N/A')} | {u.get(key, 'N/A')} |\n")

    md.append("\n## Training And Evaluation Charts\n")
    for plot in plots:
        md.append(f"![{plot}]({plot})\n")

    if q.get("example_prediction") or u.get("example_prediction"):
        md.append("\n## Example Predictions\n")
        if q.get("example_prediction"):
            md.append(f"- QLoRA prediction: {q['example_prediction']}\n")
        if u.get("example_prediction"):
            md.append(f"- Unsloth prediction: {u['example_prediction']}\n")

    md.append("\n## Observations\n")
    md.append("The benchmark reports both efficiency metrics and dataset-wide caption quality metrics on the current cleaned dataset.\n")
    if q.get("skipped_invalid_images") or u.get("skipped_invalid_images"):
        md.append(
            "The `skipped_invalid_images` metric reflects corrupt downloads found during evaluation of the saved model artifacts; those files were removed before the current dataset was rebuilt.\n"
        )

    path = f"{OUTPUT_DIR}/experiment_report.md"
    with open(path, "w") as f:
        f.writelines(md)

    print("Report generated:", path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    q, u = load_results()
    dataset_size = load_dataset_size()

    plots = []
    for metric in numeric_metrics(q, u):
        plots.append(plot_metric(metric, q.get(metric), u.get(metric)))

    generate_markdown(q, u, plots, dataset_size)


if __name__ == "__main__":
    main()
