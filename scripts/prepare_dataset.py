import os
import json
from datasets import load_dataset
from tqdm import tqdm


RAW_IMAGE_DIR = "data/raw/coco_images"
OUTPUT_FILE = "data/processed/train.json"


def main():

    os.makedirs("data/processed", exist_ok=True)

    print("Loading COCO captions dataset...")
    dataset = load_dataset("coco_captions", "2017", split="train")

    processed_data = []

    print("Formatting dataset...")

    for idx, sample in enumerate(tqdm(dataset)):

        caption = sample["captions"][0]

        image_path = os.path.join(RAW_IMAGE_DIR, f"{idx}.jpg")

        if not os.path.exists(image_path):
            continue

        formatted_sample = {
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

        processed_data.append(formatted_sample)

    print("Saving processed dataset...")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(processed_data, f, indent=2)

    print(f"Dataset saved to {OUTPUT_FILE}")
    print(f"Total samples: {len(processed_data)}")


if __name__ == "__main__":
    main()