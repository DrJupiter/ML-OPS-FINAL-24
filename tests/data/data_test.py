import hydra

import tests


def test_data_shape() -> None:
    import numpy as np
    import yaml
    from datasets import load_from_disk

    from project.data import make_dataset

    # get config file (since we cant use the common hydra method as tsts cant have input)
    with open("conf/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # make dataset
    # Also tests if thing functino works
    make_dataset.process_and_save_dataset(cfg)

    # find path to the dataset so we can use it
    dataset_path = cfg["data"]["path"] + f"cifar10-{cfg['model']['name_or_path'].replace('/', '-')}"
    dataset = load_from_disk(dataset_path)

    # test if the assumed amounts images in each group is correct.
    # (N-images, [img,label]) -> (N,2)
    assert dataset["test"].shape == (10000, 2), "Error in amount of images or dataset shape"
    assert dataset["train"].shape == (40000, 2)
    assert dataset["validation"].shape == (10000, 2)

    # Test if some random images have the correct shape
    i = np.random.randint(0, 10000)
    assert dataset["test"]["img"][i].size == (32, 32), "Images of wrong shape, expected (32,32)"
    assert type(dataset["test"]["label"][i]) == int, "Labels of wrong type, expected int"
    assert dataset["test"]["img"][i].mode == "RGB", "Image is not RGB as expected"
