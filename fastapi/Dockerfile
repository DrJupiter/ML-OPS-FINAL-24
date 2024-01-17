# Base image
FROM python:3.10.10

EXPOSE $PORT
WORKDIR /fastapi

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_fastapi.txt requirements_fastapi.txt

# The --no-chache-dir flag is important to reduce the image size
RUN pip install -r requirements_fastapi.txt --no-cache-dir

COPY ./app/ ./app/
COPY ./model/ ./model/

# Naming the training script as the entrypoint for the docker image
# The "-u" here makes sure that any output from our script e.g. any print(...) statements gets redirected to our terminal
CMD exec uvicorn app.main:app --port $PORT --host 0.0.0.0 --workers 1
