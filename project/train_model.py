# Imports
# TODO Structure imports according to the thing i cant remember
# Typing
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
import yaml
from datasets import load_from_disk, load_metric
from omegaconf import DictConfig
from PIL.PngImagePlugin import PngImageFile
from transformers import EvalPrediction, Trainer, TrainingArguments, ViTFeatureExtractor
from transformers.image_processing_utils import BatchFeature

from project.models.model import get_model


### Define helper functions ###
def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """Computes accruacy metric"""
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def collater(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """The function used to form a batch from a list of elements of train_dataset or eval_dataset"""
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


def get_ViTFeatureExtractor(cfg: DictConfig) -> Callable:
    """Gets the Feature extractor"""
    return ViTFeatureExtractor.from_pretrained(cfg["model"]["name_or_path"])


def get_transform(cfg: DictConfig) -> Callable:
    """Gets the transformer function"""

    def transform(sample: Dict[str, List[PngImageFile]]) -> BatchFeature:
        """Extracts the features from the data using ViTFeatureExtractor"""
        feature_extractor = get_ViTFeatureExtractor(cfg)
        inputs = feature_extractor([x for x in sample["img"]], return_tensors="pt")
        inputs["label"] = sample["label"]
        return inputs

    return transform


### Main function ###
@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def train_model(cfg: DictConfig) -> None:
    """
        Loads/creates, trains and tests cfg model.\\
        Also loads data
    """
    # Load the dataset
    # Because the model uses "/" in its name, we need to replace it with "-" in the dataset path
    print(cfg)
    dataset_path = cfg["data"]["path"] + f"cifar10-{cfg['model']['name_or_path'].replace('/', '-')}"
    dataset = load_from_disk(dataset_path)

    # Initialize the model
    model = get_model(cfg).to(cfg["model"]["device"])

    # Initialize the feature extractor
    feature_extractor = get_ViTFeatureExtractor(cfg)

    # Transform samples using features
    transform = get_transform(cfg)

    # apply transformation above to data
    ds = dataset.with_transform(transform)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        evaluation_strategy=cfg["training"]["evaluation_strategy"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        fp16=cfg["training"]["fp16"],
        save_steps=cfg["training"]["save_steps"],
        eval_steps=cfg["training"]["eval_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        save_total_limit=cfg["training"]["save_total_limit"],
        remove_unused_columns=cfg["training"]["remove_unused_columns"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
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
        tokenizer=feature_extractor,
    )

    # Start training
    train_results = trainer.train()

    # save model
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Log metrics, save state, etc.
    metrics = trainer.evaluate(ds["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    pass
    # train_model()
