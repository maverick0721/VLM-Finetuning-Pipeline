import json
import yaml
import wandb

from datasets import Dataset

from unsloth import FastLanguageModel

from utils.vision_collator import VisionLanguageCollator


CONFIG_PATH = "configs/unsloth.yaml"
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

    wandb.init(project="vlm-unsloth-training")

    dataset = load_dataset()

    model_name = config["model"]["name"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=True
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"]
    )

    collator = VisionLanguageCollator(model_name)

    trainer = FastLanguageModel.get_trainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["epochs"]
    )

    trainer.train()

    model.save_pretrained(config["training"]["output_dir"])


if __name__ == "__main__":
    main()