import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import evaluate
import hydra
import numpy as np
import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from train_model import get_transform, get_ViTFeatureExtractor
from transformers import EvalPrediction, Trainer, TrainingArguments, ViTImageProcessor, set_seed
from transformers.image_processing_utils import BatchFeature

from models.model import get_model


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    # lis = []
    # for i,batch in enumerate(dataloader):
    #     print("going for", i)
    #     if len(batch["pixel_values"].shape) == 3: # if b=1 then we need unsqueeze to get 4 dims
    #         out = model(batch["pixel_values"].unsqueeze(0)) # apply model to datapoint
    #     else:
    #         out = model(batch["pixel_values"]) # apply model to datapoint

    #     lis.append(out.logits) # get the prediction tensor out
    #     if i==0:
    #         break

    for i, batch in enumerate(dataloader):
        print("sub loop", i)
        model(batch["pixel_values"].unsqueeze(0))

    # prof.export_chrome_trace("trace.json")

    # return torch.cat(lis, 0)


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def manual_test(cfg: DictConfig) -> None:
    with profile(
        activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/prof")
    ) as prof:
        if cfg["model"]["device"] == "cpu":
            warnings.warn("cpu is currently being used. This will result in slow training.")
        set_seed(cfg["training"]["seed"])

        dataset_path = cfg["data"]["path"] + f"cifar10-{cfg['model']['name_or_path'].replace('/', '-')}"
        dataset = load_from_disk(dataset_path)

        # Initialize the model
        model = get_model(cfg).to(cfg["model"]["device"])

        # Initialize the feature extractor
        # feature_extractor = get_ViTFeatureExtractor(cfg)

        # Transform samples using features
        transform = get_transform(cfg)

        # apply transformation above to data
        ds = dataset.with_transform(transform)

        for i, batch in enumerate(ds["test"]):
            print("main loop", i)
            if i == 10:
                break
            predict(model, [batch])
            # prof.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))


if __name__ == "__main__":
    manual_test()

    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     manual_test()

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    # prof.export_chrome_trace("trace.json")
