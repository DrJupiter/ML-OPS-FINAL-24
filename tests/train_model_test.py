import numpy as np
from torch import tensor

import tests


def test_compute_metrics():
    from datasets import load_metric
    from transformers import EvalPrediction

    from project.train_model import compute_metrics

    N = 10
    preds = np.arange(N)
    label_ids = np.zeros_like(preds)
    preds = np.stack((preds, np.zeros_like(preds))).transpose()

    ep_true = EvalPrediction(preds, label_ids)
    ep_false = EvalPrediction(preds, label_ids - 1.0)

    assert (
        compute_metrics(ep_true)["accuracy"] == 1.0
    ), "When predictions and labels are identical compute metrics should find 1.0 accuracy, it does not"
    assert (
        compute_metrics(ep_false)["accuracy"] == 0.0
    ), "When predictions and labels are fully disimilar compute metrics should find 0 accuracy, it does not"


def test_collater():
    from torch import arange, equal

    from project.train_model import collater

    N = 10
    B = 12

    batch = []
    for i in range(N):
        batch.append({"pixel_values": arange(B), "label": tensor([0])})

    collacated_batch = collater(batch)

    for i in range(N):
        assert equal(collacated_batch["pixel_values"][i], arange(B))
        assert equal(collacated_batch["labels"][i], tensor(0))


def test_get_ViTFeatureExtractor():
    # get config file (since we cant use the common hydra method as tsts cant have input)
    import yaml

    from project.train_model import get_ViTFeatureExtractor

    with open("conf/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    feature_extractor = get_ViTFeatureExtractor(cfg)

    # craete image used as input
    from PIL import Image

    im = np.ones((32, 32, 3), dtype=np.uint8)
    im = Image.fromarray(im, "RGB")

    # apply func
    out = feature_extractor([x for x in [im]], return_tensors="pt")

    # check if it correctly transofrmed the data
    assert list(out.keys()) == ["pixel_values"]
    assert out["pixel_values"].size() == (1, 3, 224, 224)  # 1 image with 3 channels and H=W=224, as the network takes


def test_get_transform():
    import yaml

    from project.train_model import get_transform

    with open("conf/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    transform = get_transform(cfg)

    # create data to test on
    from PIL import Image

    ims = [np.ones((32, 32, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8)]
    ims = [Image.fromarray(im, "RGB") for im in ims]
    sample = {"img": ims, "label": [1, 0]}

    # apply func
    out = transform(sample)

    # make sure output is as expected
    assert list(out.keys()) == ["pixel_values", "label"]
    assert out["pixel_values"].size() == (2, 3, 224, 224)  # 2 image with 3 channels and H=W=224, as the network takes
    assert out["label"] == [1, 0]  # the expected labels
