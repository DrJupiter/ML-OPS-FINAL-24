# Base image
FROM python:3.11.7-slim-bullseye
#FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir
#RUN pip install . --no-deps --no-cache-dir

#
#RUN dvc init --no-scm
#COPY .dvc/config .dvc/config
#COPY *.dvc *.dvc
#RUN dvc config core.no_scm true
#RUN dvc pull

COPY data/ data/

COPY README.md README.md

COPY project/ project/
COPY conf/ conf/


ENV PYTHONPATH "${PYTHONPATH}:/"

ENTRYPOINT ["python", "-u", "project/train_model.py"]
