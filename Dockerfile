# Dockerfile for SpaceTrack Flood
# Production reproducible container for hackathon evaluation and CI
FROM python:3.11-slim-bookworm

LABEL maintainer="SpaceTrack Flood Team"
LABEL description="Reproducible satellite flood and debris mapping pipeline"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies including GDAL, PROJ, WeasyPrint deps, and Devanagari fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b \
    libffi-dev \
    libcairo2 \
    fonts-noto-core \
    fonts-noto-extra \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source repository
COPY . /workspace

# Install package in editable mode
RUN pip install -e .

EXPOSE 8501 8000

# Default entrypoint
CMD ["python", "run.py", "--help"]
