import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_PATH = "models/qlora"

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto"
)


def generate_caption(image):

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

    return result


interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Vision Language Model Demo"
)

interface.launch()