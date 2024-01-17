import os

import hydra
import torchvision
import torchvision.transforms as transforms
import yaml
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from transformers import ViTFeatureExtractor


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


def download_and_convert_images():
    # Transform the data to torch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Download CIFAR10 dataset
    dataset_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Directories for saving images
    train_dir = "data/raw/train/"
    test_dir = "data/raw/test/"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Function to save images
    def save_images(dataset, directory):
        for i, (image, label) in enumerate(dataset):
            # Convert tensor to PIL image
            image = transforms.ToPILImage()(image)

            # Save image
            image.save(os.path.join(directory, f"{i}_label_{label}.png"))

    # Save training and test images
    save_images(dataset_train, train_dir)
    save_images(dataset_test, test_dir)


if __name__ == "__main__":
    with open("conf/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    process_and_save_dataset(config)
    download_and_convert_images()
