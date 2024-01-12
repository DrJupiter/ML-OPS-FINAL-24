from project.train_model import collater, compute_metrics, get_transform, get_ViTFeatureExtractor


def test_test() -> None:
    None


def test_2_test() -> None:
    assert True, "nein"


def import_test() -> None:
    from project.train_model import collater, compute_metrics, get_transform, get_ViTFeatureExtractor


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
