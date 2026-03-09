import os
import json
from datasets import load_dataset
from tqdm import tqdm


RAW_DIR = "data/raw/images"
OUTPUT_FILE = "data/processed/train.json"

LIMIT = 100


def main():

    os.makedirs("data/processed", exist_ok=True)

    dataset = load_dataset("conceptual_captions", split="train")

    data = []

    for idx, sample in enumerate(tqdm(dataset)):

        if idx >= LIMIT:
            break

        image_path = f"{RAW_DIR}/{idx}.jpg"

        if not os.path.exists(image_path):
            continue

        caption = sample["caption"]

        data.append({
            "image": image_path,
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
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("Dataset prepared:", OUTPUT_FILE)


if __name__ == "__main__":
    main()