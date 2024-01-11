import hydra
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from models.model import get_model
import yaml
import torch
import numpy as np
from datasets import load_metric
from transformers import ViTFeatureExtractor


def train_model(config):
    # Load the dataset
    # Because the model uses "/" in its name, we need to replace it with "-" in the dataset path
    dataset_path = config["data"]["path"] + f"cifar10-{config['model']['name_or_path'].replace('/', '-')}"
    dataset = load_from_disk(dataset_path)

    # Initialize the model
    model = get_model(config).to(config["model"]["device"])

    # Initialize the feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained(config["model"]["name_or_path"])

    def collater(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch]),
        }

    metric = load_metric("accuracy")

    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    def transform(sample):
        inputs = feature_extractor([x for x in sample["img"]], return_tensors="pt")
        inputs["label"] = sample["label"]
        return inputs

    ds = dataset.with_transform(transform)

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        num_train_epochs=config["training"]["num_train_epochs"],
        fp16=config["training"]["fp16"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        logging_steps=config["training"]["logging_steps"],
        learning_rate=config["training"]["learning_rate"],
        save_total_limit=config["training"]["save_total_limit"],
        remove_unused_columns=config["training"]["remove_unused_columns"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        overwrite_output_dir=True,
        report_to="wandb",
    )

    # Initialize the trainer with the model, training arguments, tokenizer, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collater,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=feature_extractor
    )

    # Start training
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Log metrics, save state, etc.
    metrics = trainer.evaluate(ds["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    with open("conf/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_model(config)
