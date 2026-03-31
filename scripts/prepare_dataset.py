import json
import os

from datasets import load_dataset
from tqdm import tqdm


RAW_DIR = "data/raw/images"
OUTPUT_FILE = "data/processed/train.json"
LIMIT = 100


def main():
    os.makedirs("data/processed", exist_ok=True)

    dataset = load_dataset("conceptual_captions", split=f"train[:{LIMIT}]")
    data = []

    print("Preparing dataset...")

    for idx, sample in enumerate(tqdm(dataset)):
        image_path = f"{RAW_DIR}/{idx}.jpg"
        if not os.path.exists(image_path):
            continue

        caption = sample.get("caption") or ""
        if not caption.strip():
            continue

        data.append(
            {
                "image": image_path,
                "conversation": [
                    {"from": "human", "value": "Describe the image."},
                    {"from": "assistant", "value": caption},
                ],
            }
        )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Dataset saved to: {OUTPUT_FILE} ({len(data)} samples)")
    if not data:
        raise RuntimeError("No valid samples produced. Ensure download step succeeded.")


if __name__ == "__main__":
    main()
