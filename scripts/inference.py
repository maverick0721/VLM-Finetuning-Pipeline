import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_PATH = "models/qlora"


def main():

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto"
    )

    image = Image.open("test.jpg").convert("RGB")

    prompt = "Describe the image."

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=100
    )

    result = processor.decode(output[0], skip_special_tokens=True)

    print(result)


if __name__ == "__main__":
    main()