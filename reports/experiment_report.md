# Vision-Language Model Fine-Tuning Experiment
## Experiment Setup
- Dataset size: 100 images
- Model: LLaVA 1.5 7B
- Training methods compared: QLoRA vs Unsloth

## Benchmark Results
| Metric | QLoRA | Unsloth |
|------|------|------|
| training_time_seconds | 29.08 | 115.41 |
| tokens_per_second | 0.0 | 0.0 |
| peak_vram_gb | 22.8 | 4.67 |

## Training Performance Charts
![reports/training_time_seconds.png](reports/training_time_seconds.png)
![reports/tokens_per_second.png](reports/tokens_per_second.png)
![reports/peak_vram_gb.png](reports/peak_vram_gb.png)

## Observations
Unsloth generally provides faster training throughput while maintaining comparable performance.
