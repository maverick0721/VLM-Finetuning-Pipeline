import torch
from PIL import Image
from transformers import AutoProcessor


class VisionLanguageCollator:
    def __init__(self, processor_name):
        self.processor = AutoProcessor.from_pretrained(processor_name)

    def __call__(self, batch):
        images = []
        texts = []

        for sample in batch:
            image = Image.open(sample["image"]).convert("RGB")

            conversation = sample["conversation"]

            prompt = conversation[0]["value"]
            answer = conversation[1]["value"]

            text = f"USER: {prompt}\nASSISTANT: {answer}"

            images.append(image)
            texts.append(text)

        inputs = self.processor(
            images=images,
            text=texts,
            padding=True,
            return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()

        inputs["labels"] = labels

        return inputs