import io
import os

# typing
from os import PathLike as PPath
from pathlib import Path

import torch
from google.cloud import storage
from PIL import Image
from torch.nn.functional import softmax
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    ViTForImageClassification,
    ViTImageProcessor,
)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI()
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: PPath) -> None:
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def download_directory_from_gcs(bucket_name: str, source_directory: str, destination_directory: str) -> None:
    """Download all files in the specified directory from GCS to a local directory."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=source_directory)
    for blob in blobs:
        # Construct the destination path
        destination_blob_path = Path(destination_directory) / Path(blob.name).relative_to(source_directory)
        # Make sure that the destination directory exists
        destination_blob_path.parent.mkdir(parents=True, exist_ok=True)
        # Download the blob to a local file
        download_blob(bucket_name, blob.name, destination_blob_path)


download_directory_from_gcs(
    bucket_name="ml-ops-2024-bucket",
    source_directory="project/models/trained_models/cifar10/",
    destination_directory="./model/",
)

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    inputs = image_processor(image, return_tensors="pt").to(device)
    model = ViTForImageClassification.from_pretrained("./model/").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = softmax(logits, dim=-1)

    predicted_label = logits.argmax(-1).item()
    predicted_probabilities = probabilities[0, predicted_label].item()
    prediction = model.config.id2label[predicted_label]

    content = f"""
<!DOCTYPE html>
<html>
<head>
<title>ViT CIFAR10 Classifier</title>
<style>
body, html {{
    height: 100%;
    margin: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background-color: black;
    color: white;
}}
.header {{
    margin-top: 20px;
    margin-bottom: 20px;
}}
h1 {{
    font-size: 50px;
    font-weight: bold;
}}
.centered-content {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-grow: 1;
}}
.button {{
    margin-top: 20px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    background-color: white;
    color: black;
    border: none;
    border-radius: 5px;
    text-decoration: none; /* Removes underline from anchor tag */
}}
</style>
</head>
<body>
<div class="header">
    <h1>ViT CIFAR10 Classifier</h1>
</div>
<div class="centered-content">
    <h2>Predicted Label: {prediction}</h2>
    <h2>Probability: {predicted_probabilities:.3f}</h2>
    <a href="/" class="button">Try Again</a> <!-- Button to redirect to home page -->
</div>
</body>
</html>
    """
    return HTMLResponse(content=content)


@app.get("/")
async def main():
    content = """
<!DOCTYPE html>
<html>
<head>
<title>ViT CIFAR10 Classifier</title>
<style>
body, html {
    height: 100%;
    margin: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background-color: black;
    color: white;
}
.header {
    margin-top: 20px;
    margin-bottom: 20px;
}

h1 {
    font-size: 50px;
    font-weight: bold;
}
.centered-form {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-grow: 1;
}
input[type="file"], input[type="submit"] {
    width: 80%;
    margin: 20px 0;
    display: block; /* Make input elements block-level */
}
input[type="submit"] {
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}
</style>
</head>
<body>
<div class="header">
    <h1>ViT CIFAR10 Classifier</h1>
</div>
<div class="centered-form">
    <form action="/uploadfile/" enctype="multipart/form-data" method="post">
    <input name="file" type="file">
    <input type="submit" value="Submit">
    </form>
</div>
</body>
</html>
    """
    return HTMLResponse(content=content)
