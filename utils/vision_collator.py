from PIL import Image


class VisionLanguageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = []
        prompts = []

        for sample in batch:
            try:
                image = Image.open(sample["image"]).convert("RGB")
            except Exception:
                continue

            question = sample["conversation"][0]["value"]
            answer = sample["conversation"][1]["value"]
            prompt = "<image>\n" + question + "\n" + answer

            images.append(image)
            prompts.append(prompt)

        if len(images) == 0:
            raise ValueError("All images in the batch failed to load; cannot build training batch.")

        inputs = self.processor(
            images=images,
            text=prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
