import os
from datasets import load_dataset, DatasetDict
from transformers import ViTFeatureExtractor
from omegaconf import DictConfig
import hydra
import yaml


def process_and_save_dataset(config: DictConfig):
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    processed_data_path = os.path.join(current_script_dir, "..", "..", "data", "processed")

    ds = load_dataset("cifar10")

    # Split the training dataset for validation
    split_ds = ds["train"].train_test_split(test_size=10000)

    # Create a DatasetDict without applying transformations
    prepared_ds = DatasetDict({"train": split_ds["train"], "test": ds["test"], "validation": split_ds["test"]})

    # Save the dataset to the constructed path
    save_path = os.path.join(processed_data_path, f'cifar10-{config["model"]["name_or_path"].replace("/", "-")}')
    prepared_ds.save_to_disk(save_path)


if __name__ == "__main__":
    with open("conf/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    process_and_save_dataset(config)
