import json
import os

try:
    import unsloth
    from unsloth import FastVisionModel
except Exception:
    unsloth = None
    FastVisionModel = None

import torch
import wandb
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    TrainingArguments,
)

from scripts.benchmark import BenchmarkTracker, BenchmarkTrainer, save_results
from utils.vision_collator import VisionLanguageCollator


CONFIG_PATH = "configs/experiment.yaml"
DATA_PATH = "data/processed/train.json"
OUTPUT_DIR = "models/unsloth"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_dataset():
    with open(DATA_PATH) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def setup_wandb(config):
    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        wandb.init(project=config["project"], name="unsloth-training", config=config)
        return "wandb", True

    os.environ["WANDB_MODE"] = "disabled"
    print("WANDB_API_KEY not set. Running training without Weights & Biases logging.")
    return "none", False


def load_unsloth_model(model_name, config):
    try:
        if FastVisionModel is None:
            raise RuntimeError("Unsloth is not available in this environment.")
        print("Loading Unsloth FastVisionModel...")

        model, processor = FastVisionModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
            device_map="auto",
        )

        model = FastVisionModel.get_peft_model(
            model,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=["q_proj", "v_proj"],
        )

        return model, processor
    except Exception as exc:
        print(f"Unsloth vision path unavailable ({exc}). Falling back to HF vision LoRA setup.")

        processor = AutoProcessor.from_pretrained(model_name)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        return model, processor


def main():
    config = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_to, wandb_enabled = setup_wandb(config)

    dataset = load_dataset()
    model_name = config["model"]["name"]

    model, processor = load_unsloth_model(model_name, config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=float(config["training"]["learning_rate"]),
        logging_steps=5,
        save_strategy="epoch",
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        bf16=True,
        fp16=False,
    )

    tracker = BenchmarkTracker()

    trainer = BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VisionLanguageCollator(processor),
        tracker=tracker,
    )


    tracker.start()
    trainer.train()
    tracker.stop()

    results = tracker.results()
    save_results(results, f"{OUTPUT_DIR}/benchmark.json")

    if wandb_enabled:
        wandb.log(results)

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("Unsloth training complete")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
