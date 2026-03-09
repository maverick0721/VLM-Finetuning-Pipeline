import os
import json
from datasets import load_dataset
from tqdm import tqdm


OUTPUT_PATH = "data/processed/train.json"


def format_sample(sample):
    caption = sample["captions"][0]

    return {
        "image": sample["image"].filename if hasattr(sample["image"], "filename") else "",
        "conversation": [
            {
                "from": "human",
                "value": "Describe the image in detail."
            },
            {
                "from": "assistant",
                "value": caption
            }
        ]
    }


def main():

    os.makedirs("data/processed", exist_ok=True)

    dataset = load_dataset("coco_captions", "2017")

    formatted = []

    for sample in tqdm(dataset["train"]):
        formatted.append(format_sample(sample))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(formatted, f)

    print("Dataset prepared:", OUTPUT_PATH)


if __name__ == "__main__":
    main()