import numpy as np

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

    print("##", compute_metrics(ep_true)["accuracy"])
    print("##", compute_metrics(ep_false)["accuracy"])

    assert (
        compute_metrics(ep_true)["accuracy"] == 1.0
    ), "When predictions and labels are identical compute metrics should find 1.0 accuracy, it does not"
    assert (
        compute_metrics(ep_false)["accuracy"] == 0.0
    ), "When predictions and labels are fully disimilar compute metrics should find 0 accuracy, it does not"


def test_collater():
    from project.train_model import collater

    None


def test_get_ViTFeatureExtractor():
    from project.train_model import get_ViTFeatureExtractor

    None


def test_get_transform():
    from project.train_model import get_transform

    None


# project\train_model.py


# def test_compute_metrics():
#     from transformers import EvalPrediction
#     ep = EvalPrediction()

#     compute_metrics(ep)


# test_compute_metrics()


# ### Define helper functions ###
# def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
#     """Computes accruacy metric"""
#     metric = load_metric("accuracy")
#     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


# def collater(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#     """The function used to form a batch from a list of elements of train_dataset or eval_dataset"""
#     return {
#         "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
#         "labels": torch.tensor([x["label"] for x in batch]),
#     }


# def get_ViTFeatureExtractor(cfg: DictConfig) -> Callable:
#     """Gets the Feature extractor"""
#     return ViTFeatureExtractor.from_pretrained(cfg["model"]["name_or_path"])


# def get_transform(cfg: DictConfig) -> Callable:
#     """Gets the transformer function"""

#     def transform(sample: Dict[str, List[PngImageFile]]) -> BatchFeature:
#         """Extracts the features from the data using ViTFeatureExtractor"""
#         feature_extractor = get_ViTFeatureExtractor(cfg)
#         inputs = feature_extractor([x for x in sample["img"]], return_tensors="pt")
#         inputs["label"] = sample["label"]
#         return inputs

#     return transform
