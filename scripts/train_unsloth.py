import json
import yaml
import wandb
import torch

from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer

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


def main():

    config = load_config()

    wandb.init(
        project=config["project"],
        name="unsloth-training",
        config=config
    )

    dataset = load_dataset()

    model_name = config["model"]["name"]

    print("Loading Unsloth optimized model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map="auto"
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=["q_proj", "v_proj"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"]
    )

    collator = VisionLanguageCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir="models/unsloth",
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        logging_steps=5,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_drop_last=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )

    tracker = BenchmarkTracker()
    tracker.start()

    trainer.train()

    tracker.stop()

    results = tracker.results()

    save_results(results, "models/unsloth/benchmark.json")

    wandb.log(results)

    model.save_pretrained("models/unsloth")
    tokenizer.save_pretrained("models/unsloth")

    print("Unsloth training complete")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()