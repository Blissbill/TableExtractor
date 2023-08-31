FROM python:3.11
WORKDIR /app
ENV POETRY_VERSION=1.5.1
RUN pip install "poetry==$POETRY_VERSION"
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install tesseract-ocr -y
RUN apt-get install tesseract-ocr-[rus] -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /app/