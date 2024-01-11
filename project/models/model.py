import hydra
import yaml
from datasets import load_from_disk
from omegaconf import DictConfig
from transformers import ViTForImageClassification


def get_model(config):
    ds = load_from_disk(config["data"]["path"] + f"cifar10-{config['model']['name_or_path'].replace('/', '-')}")
    labels = ds["train"].features["label"].names

    model = ViTForImageClassification.from_pretrained(
        config["model"]["name_or_path"],
        num_labels=len(labels),
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: str(i) for i, label in enumerate(labels)},
    ).to(config["model"]["device"])
    return model


if __name__ == "__main__":
    with open("conf/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    get_model(config)
