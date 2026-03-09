import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset


RAW_IMAGE_DIR = "data/raw/coco_images"
OUTPUT_DIR = "data/processed"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.json")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.json")

TRAIN_SPLIT = 0.95
SEED = 42


def create_sample(image_path, caption):
    return {
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
    }


def main():

    print("Preparing multimodal dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(SEED)

    dataset = load_dataset("coco_captions", "2017", split="train", streaming=False)

    samples = []

    print("Formatting samples...")

    for idx, sample in enumerate(tqdm(dataset)):

        caption = sample["captions"][0]

        image_path = os.path.join(RAW_IMAGE_DIR, f"{idx}.jpg")

        if not os.path.exists(image_path):
            continue

        formatted = create_sample(image_path, caption)

        samples.append(formatted)

    print(f"Total valid samples: {len(samples)}")

    random.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_SPLIT)

    train_data = samples[:split_idx]
    val_data = samples[split_idx:]

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    print("Saving datasets...")

    with open(TRAIN_FILE, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(VAL_FILE, "w") as f:
        json.dump(val_data, f, indent=2)

    print("Dataset preparation complete.")
    print("Saved files:")
    print(TRAIN_FILE)
    print(VAL_FILE)


if __name__ == "__main__":
    main()