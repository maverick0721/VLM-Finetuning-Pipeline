# scripts/evaluate.py

import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

from utils.metrics import compute_metrics


MODEL_PATH = "models/qlora"
DATA_PATH = "data/processed/train.json"


def load_dataset():

    with open(DATA_PATH) as f:
        data = json.load(f)

    return Dataset.from_list(data[:200])


def main():

    dataset = load_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto"
    )

    predictions = []
    references = []

    for sample in tqdm(dataset):

        prompt = sample["conversation"][0]["value"]
        answer = sample["conversation"][1]["value"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(pred)
        references.append(answer)

    metrics = compute_metrics(predictions, references)

    print("\nEvaluation Results")
    print(metrics)


if __name__ == "__main__":
    main()