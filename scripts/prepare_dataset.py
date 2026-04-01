import json
import os

from PIL import Image, UnidentifiedImageError
from datasets import load_dataset
from tqdm import tqdm


RAW_DIR = "data/raw/images"
OUTPUT_FILE = "data/processed/train.json"
LIMIT = 100


def is_valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def main():
    os.makedirs("data/processed", exist_ok=True)

    dataset = load_dataset("conceptual_captions", split=f"train[:{LIMIT}]")
    data = []
    skipped_invalid_images = 0

    print("Preparing dataset...")

    for idx, sample in enumerate(tqdm(dataset)):
        image_path = f"{RAW_DIR}/{idx}.jpg"
        if not os.path.exists(image_path):
            continue

        if not is_valid_image(image_path):
            skipped_invalid_images += 1
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

    print(f"Dataset saved to: {OUTPUT_FILE} ({len(data)} samples, skipped_invalid_images={skipped_invalid_images})")
    if not data:
        raise RuntimeError("No valid samples produced. Ensure download step succeeded.")


if __name__ == "__main__":
    main()
