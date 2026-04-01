import argparse
import json
import os
import time

try:
    import unsloth
    from unsloth import FastVisionModel
except Exception:
    unsloth = None
    FastVisionModel = None

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoProcessor

from utils.metrics import compute_metrics

try:
    from transformers import AutoModelForVision2Seq as VisionModelLoader
except ImportError:
    from transformers import LlavaForConditionalGeneration as VisionModelLoader


DEFAULT_DATA_PATH = "data/processed/train.json"
DEFAULT_OUTPUT_NAME = "evaluation.json"


def get_model_device(model):
    return next(model.parameters()).device


def load_model(model_path):
    print(f"Loading model from {model_path}...")

    if "unsloth" in model_path:
        try:
            if FastVisionModel is None:
                raise RuntimeError("Unsloth is not available in this environment.")
            model, processor = FastVisionModel.from_pretrained(
                model_name=model_path,
                load_in_4bit=True,
                device_map="auto",
            )
            return model, processor
        except Exception as exc:
            print(f"Unsloth loader failed ({exc}). Falling back to standard loader.")

    processor = AutoProcessor.from_pretrained(model_path)
    model = VisionModelLoader.from_pretrained(model_path, device_map="auto")

    return model, processor


def is_valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def load_samples(data_path, max_samples=None):
    with open(data_path) as f:
        data = json.load(f)

    samples = []
    skipped_invalid_images = 0

    for sample in data:
        image_path = sample.get("image")
        conversation = sample.get("conversation") or []
        if len(conversation) < 2 or not image_path or not os.path.exists(image_path):
            continue

        if not is_valid_image(image_path):
            skipped_invalid_images += 1
            continue

        prompt = conversation[0].get("value") or "Describe the image."
        reference = conversation[1].get("value") or ""
        if not reference.strip():
            continue

        samples.append(
            {
                "image": image_path,
                "prompt": prompt,
                "reference": reference,
            }
        )

    if max_samples is not None:
        samples = samples[:max_samples]

    if not samples:
        raise RuntimeError("No valid evaluation samples were found.")

    return samples, skipped_invalid_images


def generate_caption(model, processor, image_path, prompt, max_new_tokens):
    image = Image.open(image_path).convert("RGB")
    full_prompt = f"<image>\n{prompt}"

    inputs = processor(images=image, text=full_prompt, return_tensors="pt")
    inputs = {k: v.to(get_model_device(model)) for k, v in inputs.items()}

    start_time = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    duration = time.perf_counter() - start_time

    prompt_tokens = inputs["input_ids"].shape[1]
    generated_tokens = output[:, prompt_tokens:]
    decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return decoded.strip(), duration


def save_results(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.model, DEFAULT_OUTPUT_NAME)

    model, processor = load_model(args.model)
    samples, skipped_invalid_images = load_samples(args.data, max_samples=args.max_samples)

    predictions = []
    references = []
    total_generation_seconds = 0.0

    print(f"Evaluating {len(samples)} samples...")
    for sample in tqdm(samples):
        prediction, duration = generate_caption(
            model,
            processor,
            sample["image"],
            sample["prompt"],
            args.max_new_tokens,
        )
        predictions.append(prediction)
        references.append(sample["reference"])
        total_generation_seconds += duration

    metrics = compute_metrics(predictions, references)
    results = {
        "samples_evaluated": len(samples),
        "skipped_invalid_images": skipped_invalid_images,
        "avg_generation_seconds": round(total_generation_seconds / len(samples), 4),
        "total_generation_seconds": round(total_generation_seconds, 2),
        **metrics,
        "example_prediction": predictions[0],
        "example_reference": references[0],
    }

    print("\nEvaluation Results\n")
    for key, value in results.items():
        if key.startswith("example_"):
            continue
        print(f"{key}: {value}")

    print("\nExample Prediction:\n")
    print(results["example_prediction"])
    print("\nExample Reference:\n")
    print(results["example_reference"])

    save_results(output_path, results)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
