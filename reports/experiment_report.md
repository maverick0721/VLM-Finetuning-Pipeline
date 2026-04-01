# Vision-Language Model Fine-Tuning Experiment
## Experiment Setup
- Current cleaned dataset size: 65 samples
- Model: LLaVA 1.5 7B
- Training methods compared: QLoRA vs Unsloth

## Aggregate Results
| Metric | QLoRA | Unsloth |
|------|------|------|
| avg_generation_seconds | 2.2602 | 3.7867 |
| bleu | 0.0059 | 0.0082 |
| exact_match | 0.0 | 0.0 |
| normalized_exact_match | 0.0 | 0.0 |
| peak_vram_gb | 12.41 | 4.68 |
| rouge1 | 0.1152 | 0.1186 |
| rouge2 | 0.0154 | 0.0191 |
| rougeL | 0.0963 | 0.1024 |
| samples_evaluated | 65 | 65 |
| skipped_invalid_images | 0 | 0 |
| tokens_per_second | 1245.66 | 452.72 |
| total_generation_seconds | 146.91 | 246.13 |
| training_time_seconds | 30.74 | 84.59 |

## Training And Evaluation Charts
![reports/avg_generation_seconds.png](reports/avg_generation_seconds.png)
![reports/bleu.png](reports/bleu.png)
![reports/exact_match.png](reports/exact_match.png)
![reports/normalized_exact_match.png](reports/normalized_exact_match.png)
![reports/peak_vram_gb.png](reports/peak_vram_gb.png)
![reports/rouge1.png](reports/rouge1.png)
![reports/rouge2.png](reports/rouge2.png)
![reports/rougeL.png](reports/rougeL.png)
![reports/samples_evaluated.png](reports/samples_evaluated.png)
![reports/skipped_invalid_images.png](reports/skipped_invalid_images.png)
![reports/tokens_per_second.png](reports/tokens_per_second.png)
![reports/total_generation_seconds.png](reports/total_generation_seconds.png)
![reports/training_time_seconds.png](reports/training_time_seconds.png)

## Example Predictions
- QLoRA prediction: The image depicts a bustling bus station with numerous buses parked in a parking lot. The buses are of various sizes and are scattered throughout the scene. There are also several people walking around the station, likely
- Unsloth prediction: The image depicts a bustling bus station with numerous buses and people. There are at least 13 buses of various sizes and colors, parked in a parking lot or on the street. The buses

## Observations
The benchmark reports both efficiency metrics and dataset-wide caption quality metrics on the current cleaned dataset.
