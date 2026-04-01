import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "experiment_report.md"
PDF_PATH = ROOT / "reports" / "experiment_report.pdf"
DIAGRAM_PATH = ROOT / "reports" / "pipeline_diagram.png"
QLORA_DIR = ROOT / "models" / "qlora"
UNSLOTH_DIR = ROOT / "models" / "unsloth"


def run(cmd):
    print("\n" + "=" * 70)
    print("Running:", " ".join(cmd))
    print("=" * 70 + "\n")
    subprocess.run(cmd, cwd=ROOT, check=True)


def model_ready(model_dir):
    required = [
        model_dir / "adapter_config.json",
        model_dir / "benchmark.json",
        model_dir / "evaluation.json",
    ]
    return all(path.exists() for path in required)


def ensure_pipeline(skip_pipeline):
    if skip_pipeline:
        print("Skipping pipeline because --skip-pipeline was provided.")
        return

    if model_ready(QLORA_DIR) and model_ready(UNSLOTH_DIR) and REPORT_PATH.exists() and PDF_PATH.exists():
        print("Existing pipeline artifacts detected. Skipping retraining and reusing current results.")
        return

    run([sys.executable, "run_pipeline.py"])


def start_demo(model_path, port, label):
    cmd = [
        sys.executable,
        "-m",
        "scripts.demo",
        "--model",
        str(model_path.relative_to(ROOT)),
        "--port",
        str(port),
        "--label",
        label,
    ]
    print(f"Starting {label} demo on port {port}...")
    return subprocess.Popen(cmd, cwd=ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the VLM project end-to-end and launch both QLoRA and Unsloth demos.",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Reuse current artifacts and only launch the demos.",
    )
    parser.add_argument("--qlora-port", type=int, default=7860)
    parser.add_argument("--unsloth-port", type=int, default=7861)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_pipeline(args.skip_pipeline)

    qlora_proc = start_demo(QLORA_DIR, args.qlora_port, "QLoRA")
    unsloth_proc = start_demo(UNSLOTH_DIR, args.unsloth_port, "Unsloth")

    print("\nProject showcase is ready.\n")
    print(f"QLoRA demo   : http://127.0.0.1:{args.qlora_port}")
    print(f"Unsloth demo : http://127.0.0.1:{args.unsloth_port}")
    print(f"Report       : {REPORT_PATH}")
    print(f"PDF report   : {PDF_PATH}")
    print(f"Diagram      : {DIAGRAM_PATH}")
    print("\nUse these pages to explain the project:")
    print("1. Open the report to show the benchmark summary and generated charts.")
    print("2. Open both demos to compare QLoRA and Unsloth on the same uploaded image.")
    print("3. Press Ctrl+C here when you want to stop both demo servers.")

    children = [qlora_proc, unsloth_proc]

    try:
        while True:
            if any(proc.poll() is not None for proc in children):
                raise RuntimeError("One of the demo servers stopped unexpectedly.")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping demos...")
    finally:
        for proc in children:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in children:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
