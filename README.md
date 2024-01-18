[![Python application](https://github.com/DrJupiter/ML-OPS-FINAL-24/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/DrJupiter/ML-OPS-FINAL-24/actions/workflows/python-app.yml)
# Project

The Final Project for ML OPS by group 24

## Project Description

- **Overall goal of the project:**

  Implement a Vision Transformer Model (ViT) that is loaded from pretraining ('google/vit-base-patch16-224-in21k') and fine-tuned on the CIFAR10 dataset. The aim is to create a machine learning model capable of making predictions on previously unseen data.

- **Project Deployment:**

  The model will be served on a FastAPI endpoint, enabling users to upload images for predictions.
This interaction consists of uploading an image, and then having the probability and the class of the most likely class returned.

- Main framework

We will use [Huggingface’s transformers](https://huggingface.co/docs/transformers/index) as our third-party framwork. This package is responsible for the model and the training loop.

- Logging

During training and validation we use WandB to log our tests and loss.
This allows us to monitor training and testing progress easily from anywhere no matter where the process is happening.

- Version control

To version control our data (and model??) we use DVC (Data Version Control).
As such we have our data in a google bucket, where we can access it from anywhere.

To version control our code we use git through GitHub.

- Reproducibility

To make our results easily reproducible we use build a Docker image, that we and others can use to create Docker containers to execute our code.
Trigger creation???

Additionally we use config files through hydra to store our data, model, train and test configurations for ease of use and visibility.

- Planning

To distribute work effectively and keep track of what we are all doing, we are using GitHubs projects.
This allows easy distribution and awareness of what is happening.

- Code structure

Our code structure is created using CookieCutter.
The template we use is provided by the ML-OPS course [link to template](https://github.com/SkafteNicki/mlops_template).
This allows effective structuring of our code that is standardized and easily understandable for other developers.

- Coding

During coding we enforce PEP8 coding practice.
This is enforced by our use of ruff.

Additionally we use mypy to make sure our typing is on point.

- Continuous integration

We do continuous integration in two main ways.
First we write unit-tests to make sure our function on the smallest level work as intended.
Secondly, we use GitHub actions to make sure our commits to main aren't flawed.
For each commit we test if the code follows PEP8 (using ruff), and if the typing is correct (using mypy).
Additionally, we make sure important files aren't pushed by mistake, by stopping commits if they are.

- Model choice

The model of choice is the Vision Transformer (ViT) model proposed in [# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
This model is a strong transformer for image classification.

- Data choice

We use the CIFAR10 dataset [CIFAR10](https://paperswithcode.com/dataset/cifar-10).
CIFAR10 contains 60k RGB images of shape 32x32.
The images belong to 10 classes of basic things like cats and cars.


## How to use
    git clone https://github.com/DrJupiter/ML-OPS-FINAL-24.git
    pip install -r requirements.txt
    dvc pull

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
# ML-OPS-FINAL-24
