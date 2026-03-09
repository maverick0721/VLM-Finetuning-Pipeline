import json
import yaml
import torch
import wandb

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model

from utils.vision_collator import VisionLanguageCollator


CONFIG_PATH = "configs/qlora.yaml"
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

    wandb.init(project="vlm-qlora-training")

    dataset = load_dataset()

    model_name = config["model"]["name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    collator = VisionLanguageCollator(model_name)

    args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["epochs"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy="epoch",
        fp16=True,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator
    )

    trainer.train()

    trainer.save_model(config["training"]["output_dir"])


if __name__ == "__main__":
    main()