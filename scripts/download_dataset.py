import os
import requests
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "data/raw/images"


def download_image(url, path):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
    except:
        pass


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading Conceptual Captions dataset...")

    dataset = load_dataset("conceptual_captions", split="train")

    for idx, sample in enumerate(tqdm(dataset)):

        url = sample["image_url"]
        path = os.path.join(OUTPUT_DIR, f"{idx}.jpg")

        download_image(url, path)

        if idx > 5000:
            break

    print("Images saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()