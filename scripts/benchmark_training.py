import time
import torch

def measure_training_speed(trainer, steps=50):

    print("Running benchmark...")

    start = time.time()

    trainer.train(resume_from_checkpoint=False)

    end = time.time()

    elapsed = end - start

    tokens_per_sec = steps / elapsed

    print("Benchmark Results")
    print("------------------")
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    print(f"Tokens/sec (approx): {tokens_per_sec:.2f}")

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM Usage: {vram:.2f} GB")