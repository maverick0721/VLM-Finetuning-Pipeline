import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image


OUTPUT_DIR = "data/raw/coco_images"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset("coco_captions", "2017", split="train")

    for idx, sample in enumerate(tqdm(dataset)):

        image = sample["image"]

        path = os.path.join(OUTPUT_DIR, f"{idx}.jpg")

        image.save(path)


if __name__ == "__main__":
    main()