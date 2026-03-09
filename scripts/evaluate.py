import argparse
from PIL import Image
import torch

from transformers import AutoProcessor, LlavaForConditionalGeneration
from unsloth import FastLanguageModel


def load_model(model_path):

    if "unsloth" in model_path:

        print("Loading Unsloth model...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            device_map="auto"
        )

        processor = AutoProcessor.from_pretrained(model_path)

        return model, processor

    else:

        print("Loading standard model...")

        processor = AutoProcessor.from_pretrained(model_path)

        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto"
        )

        return model, processor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default="data/raw/images/0.jpg")

    args = parser.parse_args()

    model, processor = load_model(args.model)

    image = Image.open(args.image).convert("RGB")

    prompt = "<image>\nDescribe the image."

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

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

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()