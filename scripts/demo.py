import gradio as gr


def generate(prompt):

    return "Demo output for prompt: " + prompt


interface = gr.Interface(
    fn=generate,
    inputs="text",
    outputs="text",
    title="Vision-Language Model Demo",
    description="Simple demo interface for the trained VLM."
)


if __name__ == "__main__":

    interface.launch(server_name="0.0.0.0", server_port=7860)