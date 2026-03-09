import time
import json
import torch
import os


class BenchmarkTracker:

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_vram = 0

    def start(self):

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.time()

    def stop(self):

        self.end_time = time.time()

        if torch.cuda.is_available():
            self.peak_vram = torch.cuda.max_memory_allocated() / 1e9

    def results(self):

        duration = self.end_time - self.start_time

        return {
            "training_time_seconds": round(duration, 2),
            "peak_vram_gb": round(self.peak_vram, 2)
        }


def save_results(results, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmark results saved:", path)