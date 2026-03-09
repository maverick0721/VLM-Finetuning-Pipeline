import os
import requests
from datasets import load_dataset
from tqdm import tqdm


OUTPUT_DIR = "data/raw/images"
LIMIT = 100


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

    dataset = load_dataset("conceptual_captions", split="train")

    for idx, sample in enumerate(tqdm(dataset)):

        if idx >= LIMIT:
            break

        url = sample["image_url"]

        path = os.path.join(OUTPUT_DIR, f"{idx}.jpg")

        download_image(url, path)


if __name__ == "__main__":
    main()