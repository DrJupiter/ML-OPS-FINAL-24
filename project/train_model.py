# Imports
# TODO Structure imports according to the thing i cant remember
# Typing
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import evaluate
import hydra
import numpy as np
import torch
import wandb
from datasets import load_from_disk, load_metric
from hydra import compose, initialize
from omegaconf import DictConfig
from PIL.PngImagePlugin import PngImageFile
from transformers import EvalPrediction, Trainer, TrainingArguments, ViTImageProcessor, set_seed
from transformers.image_processing_utils import BatchFeature

from project.models.model import get_model

# from models.model import get_model


### Define helper functions ###
def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """Computes accruacy metric"""
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def collater(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """The function used to form a batch from a list of elements of train_dataset or eval_dataset"""
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


def get_ViTFeatureExtractor(cfg: DictConfig) -> Callable:
    """Gets the Feature extractor\\
        The feature extractor transforms the images so they fit the model"""
    return ViTImageProcessor.from_pretrained(cfg["model"]["name_or_path"])


def get_transform(cfg: DictConfig) -> Callable:
    """Gets the transformer function"""

    def transform(sample: Dict[str, List[PngImageFile]]) -> BatchFeature | Dict[str, torch.Tensor]:
        """Extracts the features from the data using ViTFeatureExtractor"""
        feature_extractor = get_ViTFeatureExtractor(cfg)
        inputs = feature_extractor([x for x in sample["img"]], return_tensors="pt")
        inputs["label"] = sample["label"]
        return inputs

    return transform


### Main function ###
def train_model(save_path: None | str = None) -> None:
    """
        Loads/creates, trains and tests cfg model.\\
        Also loads data
    """
    initialize(config_path="../conf/", job_name="mlops24")
    cfg = compose(config_name="config")
    cfg["model"]["device"] = str(torch.device("cuda")) if torch.cuda.is_available() else str(torch.device("cpu"))
    print(f"Using device: {cfg['model']['device']}")
    # Load the dataset
    # Because the model uses "/" in its name, we need to replace it with "-" in the dataset path
    print(cfg)

    if cfg["model"]["device"] == "cpu":
        warnings.warn("cpu is currently being used. This will result in slow training.")
    set_seed(cfg["training"]["seed"])

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

    print(f"Saving to {cfg['training']['output_dir'] if save_path is None else save_path}")
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"] if save_path is None else save_path,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        evaluation_strategy=cfg["training"]["evaluation_strategy"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        fp16=cfg["training"]["fp16"] if torch.cuda.is_available() else False,
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


import argparse


def process_command_line_args() -> None | str:
    parser = argparse.ArgumentParser(description="Script to use wandb API key")

    # Define the wandb key argument
    parser.add_argument("wandb_key", type=str, help="Your wandb API key")

    # Define an optional argument for the path
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional save path argument. Useful when using a docker volume to later extract the model from.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Use the wandb key
    wandb_key = args.wandb_key

    # Here, you can add validation for the wandb_key if needed

    # Example usage (replace with actual wandb usage as needed)
    wandb.login(key=wandb_key)
    return args.save_path


if __name__ == "__main__":
    save_path = process_command_line_args()
    train_model(save_path=save_path)
