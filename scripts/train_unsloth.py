import json
import yaml
import wandb

from datasets import Dataset
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

from utils.vision_collator import VisionLanguageCollator
from scripts.benchmark import BenchmarkTracker, save_results


CONFIG_PATH = "configs/experiment.yaml"
DATA_PATH = "data/processed/train.json"


def load_config():

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_dataset():

    with open(DATA_PATH) as f:
        data = json.load(f)

    return Dataset.from_list(data)


class TokenCounterCallback(TrainerCallback):

    def __init__(self, tracker):
        self.tracker = tracker


    def on_step_end(self, args, state, control, **kwargs):

        inputs = kwargs.get("inputs")

        if inputs and "input_ids" in inputs:

            tokens = inputs["input_ids"].numel()

            self.tracker.add_tokens(tokens)


def main():

    config = load_config()

    wandb.init(
        project=config["project"],
        name="unsloth-training",
        config=config
    )

    dataset = load_dataset()

    model_name = config["model"]["name"]

    processor = AutoProcessor.from_pretrained(model_name)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto"
    )

    collator = VisionLanguageCollator(processor)

    training_args = TrainingArguments(
        output_dir="models/unsloth",
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        logging_steps=5,
        save_strategy="epoch",
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )

    tracker = BenchmarkTracker()

    trainer.add_callback(TokenCounterCallback(tracker))

    tracker.start()

    trainer.train()

    tracker.stop()

    results = tracker.results()

    save_results(
        results,
        "models/unsloth/benchmark.json"
    )

    wandb.log(results)

    model.save_pretrained("models/unsloth")

    print("Unsloth training complete")


if __name__ == "__main__":
    main()