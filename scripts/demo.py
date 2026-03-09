import torch
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM


QLORA_MODEL = "models/qlora"
UNSLOTH_MODEL = "models/unsloth"


def load_model(path):

    processor = AutoProcessor.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    return processor, model


print("Loading QLoRA model...")
qlora_processor, qlora_model = load_model(QLORA_MODEL)

print("Loading Unsloth model...")
unsloth_processor, unsloth_model = load_model(UNSLOTH_MODEL)


def run_model(processor, model, image, prompt):

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=120
    )

    result = processor.decode(output[0], skip_special_tokens=True)

    return result


def compare_models(image, prompt):

    qlora_output = run_model(
        qlora_processor,
        qlora_model,
        image,
        prompt
    )

    unsloth_output = run_model(
        unsloth_processor,
        unsloth_model,
        image,
        prompt
    )

    return qlora_output, unsloth_output


interface = gr.Interface(
    fn=compare_models,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(
            value="Describe the image in detail.",
            label="Prompt"
        )
    ],
    outputs=[
        gr.Textbox(label="QLoRA Output"),
        gr.Textbox(label="Unsloth Output")
    ],
    title="Vision-Language Model Comparison",
    description="Compare outputs from QLoRA and Unsloth fine-tuned models."
)


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)