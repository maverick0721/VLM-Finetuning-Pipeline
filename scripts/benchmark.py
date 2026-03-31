import time
import json
import os
import torch
from transformers import Trainer


class BenchmarkTracker:

    def __init__(self):

        self.start_time = None
        self.end_time = None
        self.total_tokens = 0
        self.peak_vram = 0


    def start(self):

        self.start_time = time.time()

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()


    def add_tokens(self, token_count):

        if token_count is not None:
            self.total_tokens += int(token_count)


    def stop(self):

        self.end_time = time.time()

        if torch.cuda.is_available():

            self.peak_vram = torch.cuda.max_memory_allocated() / 1e9


    def results(self):

        duration = 0

        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time

        tokens_per_sec = 0

        if duration > 0:
            tokens_per_sec = self.total_tokens / duration

        results = {
            "training_time_seconds": round(duration, 2),
            "tokens_per_second": round(tokens_per_sec, 2),
            "peak_vram_gb": round(self.peak_vram, 2)
        }

        print("\nBenchmark Results")
        print(results)

        return results


def save_results(results, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmark results saved:", path)


def count_input_tokens(inputs):

    attention_mask = inputs.get("attention_mask")

    if attention_mask is not None:
        return int(attention_mask.sum().item())

    input_ids = inputs.get("input_ids")

    if input_ids is not None:
        return int(input_ids.numel())

    return 0


class BenchmarkTrainer(Trainer):

    def __init__(self, *args, tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker


    def training_step(self, model, inputs, num_items_in_batch=None):

        if self.tracker is not None:
            self.tracker.add_tokens(count_input_tokens(inputs))

        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)