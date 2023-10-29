FROM nvidia/cuda:11.7.0-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes \
    python3 \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    python3-tk \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./

COPY row_readme.md ./

COPY ./src ./

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "./main.py"]
