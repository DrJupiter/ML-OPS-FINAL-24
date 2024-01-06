# project

The final project for ml ops by group 24

## The first project days is all about getting started on the projects and formulating exactly what you want to work on as a group.

### 1. Start by brainstorming projects! Try to figure out exactly what you want to work with and begin to investigate what third party package that can support the project.

### 2. When you have come up with an idea, write a project description. The description is the delivery for today and should be at least 300 words. Try to answer the following questions in the description:
    - Overall goal of the project
    - What framework are you going to use and you do you intend to include the framework into your project?
    - What data are you going to run on (initially, may change)
    - What models do you expect to use

### 3. (Optional) If you want to think more about the product design of your project, feel free to fill out the ML canvas (or part of it). You can read more about the different fields on canvas here.

### 4. After having done the product description, you can start on the actual coding of the project. In the next section, a to-do list is attached that summaries what we are doing in the course. You are NOT expected to fulfill all bullet points from week 1 today.

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
