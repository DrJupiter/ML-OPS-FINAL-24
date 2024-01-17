#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <wandb_key>"
    exit 1
fi

WANDB_KEY=$1

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# PULL THE IMAGE
sudo docker pull drjupiter/mlops24:latest-gpu

# CREATE A VOLUME FOR EXTERNALLY SAVING TO
sudo docker volume create drjupiter_trained_models

# RUN THE IMAGE
sudo docker run -v drjupiter_trained_models:/trained_model --gpus all drjupiter/mlops24:latest-gpu --save_path /trained_model $WANDB_KEY
