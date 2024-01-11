# Base image
FROM python:3.10.10

ENV WANDB_API_KEY=************
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_docker_train.txt requirements_docker_train.txt
COPY pyproject.toml pyproject.toml
COPY project/ project/
COPY data/ data/
COPY conf/ conf/


# upgrade pip
RUN pip install --upgrade pip

# Set working directory and install requirements
# The --no-chache-dir flag is important to reduce the image size
WORKDIR /
RUN pip install -r requirements_docker_train.txt

# Naming the training script as the entrypoint for the docker image
# The "-u" here makes sure that any output from our script e.g. any print(...) statements gets redirected to our terminal
ENTRYPOINT ["python", "-u", "project/train_model.py"]

