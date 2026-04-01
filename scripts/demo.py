import argparse

import gradio as gr
import torch
from PIL import Image

from scripts.evaluate import get_model_device, load_model


FALLBACK_PROMPTS = [
    "Describe the image.",
    "Describe this image in one sentence.",
    "Caption this image.",
]


def run_generation(model, processor, image, prompt):
    full_prompt = "<image>\n" + prompt
    inputs = processor(
        images=image,
        text=full_prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(get_model_device(model)) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    prompt_tokens = inputs["input_ids"].shape[1]
    generated_tokens = output[:, prompt_tokens:]
    decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return decoded.strip()


def is_useful_output(text, prompt):
    normalized = (text or "").strip()
    if not normalized:
        return False

    if normalized.lower() == prompt.strip().lower():
        return False

    if normalized in {"?", ".", ",", "!"}:
        return False

    return any(char.isalnum() for char in normalized)


def build_demo(model, processor, model_label):
    def generate(prompt, image):
        if image is None:
            return "Please upload an image."

        image = Image.fromarray(image).convert("RGB")

        candidate_prompts = []
        if prompt and prompt.strip():
            candidate_prompts.append(prompt.strip())

        for fallback_prompt in FALLBACK_PROMPTS:
            if fallback_prompt not in candidate_prompts:
                candidate_prompts.append(fallback_prompt)

        for candidate_prompt in candidate_prompts:
            decoded = run_generation(model, processor, image, candidate_prompt)
            if is_useful_output(decoded, candidate_prompt):
                if candidate_prompt != (prompt or "").strip():
                    return f"{decoded}\n\n[Used fallback prompt: {candidate_prompt}]"
                return decoded

        return "The model could not produce a useful caption for this image."

    return gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt", value="Describe the image."),
            gr.Image(label="Upload Image"),
        ],
        outputs=gr.Textbox(label="Model Output"),
        title=f"Vision-Language Model Demo ({model_label})",
        description=f"Interactive captioning demo for the {model_label} fine-tuned model.",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/unsloth")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--label")
    return parser.parse_args()


def main():
    args = parse_args()
    model_label = args.label or args.model.split("/")[-1].upper()

    print(f"Loading model for demo: {args.model}")
    model, processor = load_model(args.model)

    demo = build_demo(model, processor, model_label)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
