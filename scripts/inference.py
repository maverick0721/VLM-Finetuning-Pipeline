import argparse
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--image")

    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)

    model = LlavaForConditionalGeneration.from_pretrained(args.model)

    image = Image.open(args.image)

    inputs = processor(
        images=image,
        text="Describe the image",
        return_tensors="pt"
    )

    output = model.generate(**inputs)

    print(processor.decode(output[0]))


if __name__ == "__main__":
    main()