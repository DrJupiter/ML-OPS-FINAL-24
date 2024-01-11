[![Python application](https://github.com/DrJupiter/ML-OPS-FINAL-24/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/DrJupiter/ML-OPS-FINAL-24/actions/workflows/python-app.yml)
# Project

The Final Project for ML OPS by group 24

## Alternative Project Description (proposal)

- **Overall goal of the project:**

  Implement a Vision Transformer Model (ViT) that is loaded from pretraining ('google/vit-base-patch16-224-in21k') and fine-tuned on the CIFAR10 dataset. The aim is to create a machine learning model capable of making predictions on previously unseen data.

- **Project Deployment:**

  The model will be served on a FastAPI endpoint, enabling users to upload images for predictions.

## Original Project Description

- Overall goal of the project:

Implement an autoencoder on MNIST to generate images and make the training of it reproducible. We want an end-point API that allows a user to query for images. Furthermore, we want to use state-of-the-art continuous integration and continuous delivery pipelines to mimic the production requirements seen in real-world scenarios.

 -   What framework will you use and do you intend to include the framework in your project?

We will use ([Huggingface’s diffusers](https://github.com/huggingface/diffusers)) as our third-party framework. To speed up writing the training loop we will use the ([composers library](https://github.com/mosaicml/composer)). We will log our experiments in WandB.
To ensure reproducibility we will utilize the power docker to ensure software and operating system reproducibility. We will use DVC for data version control. Our projects will be configured with Python data classes. We will use Github to version control our code. Our model will be saved on hugging face.
In terms of project deployment, we will be using the FastAPI framework to serve an endpoint API that can be used for model inference by the end user. We use google cloud as our service provider.
We use a Cookiecutter template to structure our GitHub code. The template we use is provided by the ML-OPS course [link to template](https://github.com/SkafteNicki/mlops_template). This allows effective structuring of our code that is standardized and easily understandable for other developers. We use [ruff](https://github.com/astral-sh/ruff) to format our python code.

- What data are you going to run on (initially, may change)

We are going to train our model on the MNIST dataset which is a dataset of approximately 70 thousand handwritten images of the numbers 0-9. The MNIST pictures are 28 by 28 grayscale images.

- What models do you expect to use

We expect to use an autoencoder written in the third-party extension of Pytorch named Diffusers (developed by Huggingface).
We choose this framework because of prior experience with it, its good performance, and high maintenance.
More specifically we choose to work with Tiny AutoEncoder originally implemented for Stable Diffusion (TAESD). Tiny AutoEncoder was introduced in ([madebyollin/taesd](https://github.com/madebyollin/taesd)) by Ollin Boer Bohan.

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
