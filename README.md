# Machine Learning Operations Project

[![Python application](https://github.com/DrJupiter/ML-OPS-FINAL-24/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/DrJupiter/ML-OPS-FINAL-24/actions/workflows/python-app.yml)

## Overview

This project, part of DTU's "02476: Introduction to Machine Learning" course, demonstrates tools and practices for organizing, scaling, deploying, and monitoring machine learning models in research or production settings. The main focus is on hands-on experience with various frameworks, both local and cloud-based, to work with large-scale ML pipelines.

## Workflow Diagram
![MLOPS_diagram](./reports/figures/draw_io_total_fig_4.png)

The diagram above illustrates the interactions between different parts and services of the project to create the final product.

## Project Goals

- **Objective:** Implement a Vision Transformer Model (ViT) from pre-training (`google/vit-base-patch16-224-in21k`) and fine-tune it on the CIFAR10 dataset to make accurate predictions on new data.
- **Deployment:** Serve the model via a FastAPI endpoint, allowing users to upload images for predictions and receive the most likely class with its probability.

## Key Features

- **Q&A Section:** Answers to specific project-related questions are embedded within the [reports markdown file](./reports/README.md), facilitating auto-generation of the final report.
- **Project Checklist:** Detailed weekly checklist ensuring comprehensive project execution, from initial setup to deployment and monitoring.
- **Group Information:** Specific details about the group members and their contributions.

## Setup and Usage

1. **Clone the Repository:** 
    ```bash
    git clone https://github.com/DrJupiter/ML-OPS-FINAL-24.git
    ```
2. **Install Dependencies:** 
    ```bash
    pip install -r requirements.txt
    ```
3. **Download Data:**
    ```bash
    dvc pull
    ```
4. **Train the Model:**
    ```bash
    python project/train_model.py
    ```
5. **Run FastAPI Application:**
    ```bash
    uvicorn project.fastapi_app:app --reload
    ```

## Additional Information

For more detailed information about the course, visit the [course page](https://kurser.dtu.dk/course/2024-2025/02476).

## Contact

For any questions or further information, please refer to the project repository on GitHub.
