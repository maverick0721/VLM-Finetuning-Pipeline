import argparse

try:
    import unsloth
    from unsloth import FastVisionModel
except Exception:
    unsloth = None
    FastVisionModel = None

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForVision2Seq as VisionModelLoader
except ImportError:
    from transformers import LlavaForConditionalGeneration as VisionModelLoader


def get_model_device(model):
    return next(model.parameters()).device


def load_model(model_path):
    print(f"Loading model from {model_path}...")

    if "unsloth" in model_path:
        try:
            if FastVisionModel is None:
                raise RuntimeError("Unsloth is not available in this environment.")
            model, processor = FastVisionModel.from_pretrained(
                model_name=model_path,
                load_in_4bit=True,
                device_map="auto",
            )
            return model, processor
        except Exception as exc:
            print(f"Unsloth loader failed ({exc}). Falling back to standard loader.")

    processor = AutoProcessor.from_pretrained(model_path)
    model = VisionModelLoader.from_pretrained(model_path, device_map="auto")

    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default="data/raw/images/0.jpg")
    args = parser.parse_args()

    model, processor = load_model(args.model)

    image = Image.open(args.image).convert("RGB")
    prompt = "<image>\nDescribe the image."

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(get_model_device(model)) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    result = processor.decode(output[0], skip_special_tokens=True)

    print("\nModel output:\n")
    print(result)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
