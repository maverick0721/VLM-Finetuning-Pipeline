# Vision-Language Model Fine-Tuning Experiment
## Experiment Setup
- Dataset size: 100 images
- Model: LLaVA 1.5 7B
- Training methods compared: QLoRA vs Unsloth

## Benchmark Results
| Metric | QLoRA | Unsloth |
|------|------|------|
| peak_vram_gb | 12.41 | 4.68 |
| tokens_per_second | 1095.73 | 337.39 |
| training_time_seconds | 34.95 | 113.5 |

## Training Performance Charts
![reports/peak_vram_gb.png](reports/peak_vram_gb.png)
![reports/tokens_per_second.png](reports/tokens_per_second.png)
![reports/training_time_seconds.png](reports/training_time_seconds.png)

## Observations
Throughput and memory trends depend on hardware and model backend details.
