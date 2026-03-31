# Vision-Language Model Fine-Tuning Benchmark  
### QLoRA vs Unsloth for Multimodal Training

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![Transformers](https://img.shields.io/badge/huggingface-transformers-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
![Experiment Tracking](https://img.shields.io/badge/tracking-wandb-purple)

---

# Vision-Language Model Fine-Tuning Benchmark

This repository implements a **reproducible research pipeline** for benchmarking parameter-efficient fine-tuning methods for **Vision-Language Models (VLMs)**.

The project compares two modern training techniques:

| Method | Description |
|------|------|
| **QLoRA** | Quantized Low-Rank Adaptation using HuggingFace PEFT |
| **Unsloth** | High-performance LoRA training framework optimized for faster training |

The pipeline performs the entire experiment automatically:

- dataset preparation  
- multimodal fine-tuning  
- model evaluation  
- benchmark measurement  
- automatic report generation  
- architecture diagram generation  
- interactive demo deployment  

---

# System Architecture

![Pipeline Architecture](reports/pipeline_diagram.png)

Training workflow:

```
Dataset
   ↓
Download Script
   ↓
Dataset Preprocessing
   ↓
Training
 ├─ QLoRA
 └─ Unsloth
   ↓
Evaluation
   ↓
Benchmark Metrics
   ↓
Experiment Report
   ↓
Interactive Demo
```

---

# Benchmark Leaderboard

Results are generated automatically from training runs.

| Method | Training Time ↓ | Tokens / Second ↑ | Peak GPU VRAM ↓ |
|------|------|------|------|
| **QLoRA** | from benchmark.json | from benchmark.json | from benchmark.json |
| **Unsloth** | from benchmark.json | from benchmark.json | from benchmark.json |

---

# Training Metrics

The pipeline generates benchmark visualizations automatically.

## Training Time

![Training Time](reports/training_time_seconds.png)

## Training Throughput

![Tokens Per Second](reports/tokens_per_second.png)

## GPU Memory Usage

![Peak VRAM](reports/peak_vram_gb.png)

---

# Dataset

The experiment uses a **subset of Conceptual Captions**.

Configuration:

```
Dataset size: 100 images
Task: Image Captioning
Prompt: Describe the image
```

Dataset scripts:

```
scripts/download_dataset.py
scripts/prepare_dataset.py
```

---

# Model Architecture

The multimodal architecture is inspired by **LLaVA-style models**.

Components:

```
Vision Encoder → CLIP
Language Model → LLaMA
Adapter Method → LoRA
```

The model learns to generate text conditioned on image input.

---

# Repository Structure

```
VLM-Finetuning-Pipeline/

configs/
    experiment.yaml

data/
    raw/
    processed/

models/
    qlora/
    unsloth/

reports/
    experiment_report.md
    experiment_report.pdf
    training_time_seconds.png
    tokens_per_second.png
    peak_vram_gb.png
    pipeline_diagram.png

scripts/
    download_dataset.py
    prepare_dataset.py
    train_qlora.py
    train_unsloth.py
    evaluate.py
    benchmark.py
    generate_report.py
    generate_diagram.py
    export_report_pdf.py
    demo.py

utils/
    vision_collator.py
    metrics.py

run_pipeline.py
requirements.txt
Dockerfile
docker-compose.yml
.dockerignore
README.md
```

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional system dependencies:

```
pandoc
wkhtmltopdf
graphviz
```

---

# Running the Full Experiment

Execute the complete pipeline with one command:

```bash
python run_pipeline.py
```

The pipeline executes:

```
1. Dataset download
2. Dataset preprocessing

3. QLoRA training
4. QLoRA evaluation

5. Unsloth training
6. Unsloth evaluation

7. Benchmark comparison
8. Experiment report generation
9. Architecture diagram generation
10. PDF report export
11. Interactive demo launch
```

---

# Interactive Demo

Launch the demo:

```bash
python -m scripts.demo
```

Open the interface:

```
http://localhost:7860
```

Example prompt:

```
Describe the image
```

---

# Example Model Output

```
The image shows several buses parked in a large parking area.
The buses appear to be aligned in rows, suggesting a bus terminal or depot.
```

---

# Automated Experiment Reports

The pipeline generates experiment artifacts automatically.

```
reports/
 ├── experiment_report.md
 ├── experiment_report.pdf
 ├── training_time_seconds.png
 ├── tokens_per_second.png
 ├── peak_vram_gb.png
 └── pipeline_diagram.png
```

---

# Continuous Integration

Example GitHub Actions workflow:

```yaml
name: CI

on:
  push:
    branches: [ main ]

jobs:
  pipeline-test:

    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.10

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run dataset preparation
        run: |
          python scripts/prepare_dataset.py
```

---

# Reproducibility

The entire experiment is reproducible.

Running:

```bash
python run_pipeline.py
```

will automatically generate:

```
trained models
benchmark metrics
training charts
experiment reports
pipeline diagrams
```

---

# Future Work

Possible improvements:

- larger dataset experiments  
- distributed multi-GPU training  
- hyperparameter sweeps  
- additional PEFT methods  
- attention visualization for image grounding  

---

# License

MIT License

---

# Author

Shivang Gupta  
Indian Institute of Technology Roorkee

---

# Acknowledgements

Libraries used:

- HuggingFace Transformers  
- PyTorch  
- Unsloth  
- Weights & Biases  
- Gradio
