import os
import requests
from datasets import load_dataset
from tqdm import tqdm


OUTPUT_DIR = "data/raw/images"
LIMIT = 100


def download_image(url, path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            return True
    except Exception:
        return False

    return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset("conceptual_captions", split=f"train[:{LIMIT}]")

    print("Downloading dataset images...")
    success = 0

    for idx, sample in enumerate(tqdm(dataset)):
        url = sample["image_url"]
        path = os.path.join(OUTPUT_DIR, f"{idx}.jpg")

        if download_image(url, path):
            success += 1

    print(f"Downloaded {success}/{LIMIT} images to {OUTPUT_DIR}")
    if success == 0:
        raise RuntimeError("No images were downloaded. Check network access to image URLs.")


if __name__ == "__main__":
    main()
