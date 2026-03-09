import gradio as gr
import torch
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration


MODEL_PATH = "models/qlora"


print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto"
)


def generate(prompt, image):

    if image is None:
        return "Please upload an image."

    image = Image.fromarray(image).convert("RGB")

    full_prompt = "<image>\n" + prompt

    inputs = processor(
        images=image,
        text=full_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=80
        )

    result = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return result


demo = gr.Interface(

    fn=generate,

    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Image(label="Upload Image")
    ],

    outputs=gr.Textbox(label="Model Output"),

    title="Vision-Language Model Demo",

    description="Demo interface for the trained VLM."
)


if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=7860)