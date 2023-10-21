FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_VERSION=1.5.1
RUN apt-get update && apt install tesseract-ocr -y && apt-get install ffmpeg libsm6 libxext6 tesseract-ocr-[rus] zbar-tools -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /app/