import torch
from PIL import Image


class VisionLanguageCollator:

    def __init__(self, processor):

        self.processor = processor


    def __call__(self, batch):

        images = []
        prompts = []

        for sample in batch:

            image = Image.open(sample["image"]).convert("RGB")

            prompt = sample["conversation"][0]["value"]
            answer = sample["conversation"][1]["value"]

            full_text = prompt + " " + answer

            images.append(image)
            prompts.append(full_text)

        inputs = self.processor(
            images=images,
            text=prompts,
            padding=True,
            return_tensors="pt"
        )

        inputs["labels"] = inputs["input_ids"].clone()

        return inputs