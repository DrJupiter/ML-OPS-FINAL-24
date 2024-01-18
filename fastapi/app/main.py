import asyncio
import io
import os
import random
import warnings

# typing
from os import PathLike as PPath
from pathlib import Path

import pandas as pd
import torch
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from google.cloud import storage
from PIL import Image
from torch.nn.functional import softmax
from transformers import AutoFeatureExtractor, ViTForImageClassification, ViTModel, logging

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set transformers' logging to error only
logging.set_verbosity_error()

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Set up security
security = HTTPBearer()

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reference_data_path = "data/processed/embeddings_reference_data/embeddings_dataframe.pkl"
inference_data_path = "data/processed/embeddings_inference_data/embeddings_inference_dataframe.pkl"
embeddings_dir = "data/processed/embeddings_inference_data/"
pickle_filename = "embeddings_inference_dataframe.pkl"
bucket_name = "ml-ops-2024-bucket"

# Initialize Google Cloud Storage Client
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)


def load_embeddings_dataframe():
    """Load the existing embeddings dataframe from GCS bucket."""
    blob = bucket.blob(f"{embeddings_dir}{pickle_filename}")
    if blob.exists(storage_client):
        return pd.read_pickle(io.BytesIO(blob.download_as_bytes()))
    else:
        return pd.DataFrame(columns=["embeddings", "target"])


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "ML_OPS_2024-JAK":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return token


async def download_blob_async(bucket_name: str, source_blob_name: str, destination_file_name: Path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    await asyncio.to_thread(blob.download_to_filename, destination_file_name)


async def download_specific_files_async(
    bucket_name: str, source_directory: str, destination_directory: str, file_names: list
):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=source_directory)

    tasks = []
    for blob in blobs:
        if any(blob.name.endswith(f"{source_directory}{file_name}") for file_name in file_names):
            destination_blob_path = Path(destination_directory) / Path(blob.name).relative_to(source_directory)
            destination_blob_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(download_blob_async(bucket_name, blob.name, destination_blob_path))

    await asyncio.gather(*tasks)


def background_download_task(
    bucket_name: str,
    source_directory: str,
    destination_directory: str,
    file_names: list,
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(
        download_specific_files_async, bucket_name, source_directory, destination_directory, file_names
    )


async def extract_embeddings_and_save(file_contents, background_tasks: BackgroundTasks, prediction: str):
    image = Image.open(io.BytesIO(file_contents))

    # Process the image for embeddings
    image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # Load the feature model and extract embeddings
    feature_model = ViTModel.from_pretrained("./model/").to(device)
    with torch.no_grad():
        embeddings = feature_model(**inputs).last_hidden_state.squeeze().cpu().numpy().flatten()
    # Load existing DataFrame from GCS
    df = load_embeddings_dataframe()

    # Append new data
    new_data = {"embeddings": embeddings.tolist(), "target": prediction}
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    # Save updated DataFrame back to GCS
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    buffer.seek(0)
    bucket.blob(f"{embeddings_dir}{pickle_filename}").upload_from_file(buffer, content_type="application/octet-stream")


@app.post("/uploadfile/")
async def create_upload_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Allowed MIME types for images
    allowed_mime_types = ["image/jpeg", "image/png", "image/gif"]

    # Check if the uploaded file is an image
    if file.content_type not in allowed_mime_types:
        # Redirect to the home page with an error message
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error_message": "File type not allowed. Please upload an image (jpeg, png, or gif)."},
        )

    # Read the contents of the file
    contents = await file.read()

    # Open the image using PIL
    image = Image.open(io.BytesIO(contents))
    print(image.mode)

    # Handle the image based on its mode (channels)
    if image.mode == "RGB":
        pass  # Process normally
    elif image.mode == "L" or image.mode == "LA":
        image = image.convert("RGB")
    else:
        # Redirect to the home page with an error message
        return templates.TemplateResponse(
            "index.html", {"request": request, "error_message": "Unsupported image format."}
        )

    image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # For Prediction
    classification_model = ViTForImageClassification.from_pretrained("./model/").to(device)
    with torch.no_grad():
        logits = classification_model(**inputs).logits
        probabilities = softmax(logits, dim=-1)
        predicted_label = logits.argmax(-1).item()
        predicted_probabilities = probabilities[0, predicted_label].item()
        prediction = classification_model.config.id2label[predicted_label]

    # Add the task of extracting embeddings and saving to GCS to background tasks
    background_tasks.add_task(extract_embeddings_and_save, contents, background_tasks, predicted_label)

    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "prediction": prediction,
            "probability": predicted_probabilities,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/update-model")
async def update_model(background_tasks: BackgroundTasks, token: str = Depends(get_current_user)):
    background_download_task(
        "ml-ops-2024-bucket",
        "project/models/trained_models/cifar10/",
        "./model/",
        ["model.safetensors", "config.json"],
        background_tasks,
    )
    return {"message": "Model update process initiated in the background."}


async def update_reference_data():
    storage_client = storage.Client()
    bucket_name = "ml-ops-2024-bucket"
    bucket = storage_client.bucket(bucket_name)
    train_dir = "data/raw/train/"
    destination_dir = "data/processed/embeddings_reference_data/"

    blobs = storage_client.list_blobs(bucket, prefix=train_dir)
    filenames = [blob.name for blob in blobs if not blob.name.endswith("/")]
    selected_files = random.sample(filenames, min(len(filenames), 500))

    image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    feature_model = ViTModel.from_pretrained("./model/")
    feature_model.eval().to(device)

    embeddings_list = []
    labels_list = []

    for filename in selected_files:
        blob = bucket.blob(filename)
        image_bytes = await asyncio.to_thread(blob.download_as_bytes)
        image = Image.open(io.BytesIO(image_bytes))

        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = feature_model(**inputs).last_hidden_state.squeeze().cpu().numpy().flatten()
        label = filename.split("_")[-1].split(".")[0]
        embeddings_list.append(embeddings)
        labels_list.append(label)

    df = pd.DataFrame({"embeddings": embeddings_list, "target": labels_list})
    pickle_filename = "embeddings_dataframe.pkl"
    df.to_pickle(pickle_filename)

    blob = bucket.blob(f"{destination_dir}{pickle_filename}")
    await asyncio.to_thread(blob.upload_from_filename, pickle_filename)


@app.post("/update_reference_data")
async def process_embeddings(background_tasks: BackgroundTasks, token: str = Depends(get_current_user)):
    background_tasks.add_task(update_reference_data)
    return {"message": "Initiating reference data update and upload to database"}


@app.post("/update_monitoring")
async def update_monitoring(background_tasks: BackgroundTasks, token: str = Depends(get_current_user)):
    background_tasks.add_task(update_monitoring_data)
    return {"message": "Initiating monitoring update and upload to database"}


@app.get("/monitoring")
async def monitoring(request: Request):
    file_path = "monitoring.html"
    new_file_path = "app/templates/monitoring.html"

    # Check if the original monitoring.html exists
    if os.path.exists(file_path):
        # Read the original content of the file
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.readlines()
    else:
        # If the file does not exist, start with an empty content
        content = []

    # Modify the content
    new_content = []
    start_replaced = False
    for line in content:
        if "<html>" in line or "<head>" in line or '<meta charset="utf-8">' in line:
            if not start_replaced:
                new_content.append('{% extends "base.html" %}\n{% block content %}\n')
                start_replaced = True
        else:
            new_content.append(line)

    # Add end block if the start block was added
    if start_replaced:
        new_content.append("{% endblock %}\n")
    else:
        # If the original file didn't exist or didn't contain the specified tags
        new_content = [
            '{% extends "base.html" %}\n',
            "{% block content %}\n",
            "<h2>The Evidently Data Drift Report has not been created yet.</h2>",
            "<p>The report can be requested at the /update_monitoring endpoint.</p>" "{% endblock %}\n",
        ]

    # Ensure the 'app/templates' directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Write the modified/new content to the new file
    with open(new_file_path, "w+", encoding="utf-8") as file:
        file.writelines(new_content)

    return templates.TemplateResponse("monitoring.html", {"request": request})


async def update_monitoring_data():
    # Google Cloud Storage Client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Download reference data
    reference_blob = bucket.blob(reference_data_path)
    reference_buffer = io.BytesIO()
    reference_blob.download_to_file(reference_buffer)
    reference_buffer.seek(0)
    reference_data = pd.read_pickle(reference_buffer)

    # Download inference data
    inference_blob = bucket.blob(inference_data_path)
    inference_buffer = io.BytesIO()
    inference_blob.download_to_file(inference_buffer)
    inference_buffer.seek(0)
    inference_data = pd.read_pickle(inference_buffer)

    temp_inference = []
    temp_reference = []
    num_embeddings = 197

    for i in range(num_embeddings):
        temp_inference.append(inference_data["embeddings"].apply(lambda x: x[i]).to_frame(f"emb{i}"))
        temp_reference.append(reference_data["embeddings"].apply(lambda x: x[i]).to_frame(f"emb{i}"))

    # Concatenate all the new columns to the original DataFrames
    inference_data = pd.concat([inference_data] + temp_inference, axis=1)
    reference_data = pd.concat([reference_data] + temp_reference, axis=1)

    # Drop the 'embeddings' column
    inference_data.drop(columns=["embeddings"], inplace=True)
    reference_data.drop(columns=["embeddings"], inplace=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=inference_data)
    report.save_html("monitoring.html")
