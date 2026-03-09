import argparse
from PIL import Image

from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model"
    )

    parser.add_argument(
        "--image",
        default="data/raw/images/0.jpg"
    )

    args = parser.parse_args()

    print("Loading model...")

    processor = AutoProcessor.from_pretrained(args.model)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto"
    )

    image = Image.open(args.image).convert("RGB")

    prompt = "Describe the image."

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=50
    )

    result = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nModel output:\n")
    print(result)


if __name__ == "__main__":
    main()