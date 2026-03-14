# Dockerfile for torch-spyre
# Builds a container image with torch-spyre and all dependencies

FROM python:3.11-slim

LABEL maintainer="torch-spyre team"
LABEL description="Container image for torch-spyre with PyTorch and testing dependencies"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies and torch-spyre
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "torch~=2.10.0" \
    build \
    cmake \
    expecttest \
    hypothesis \
    jinja2 \
    ninja \
    numpy \
    pyyaml \
    regex \
    setuptools \
    wheel && \
    pip install --no-cache-dir -e . --no-deps --no-build-isolation

# Set default command
CMD ["/bin/bash"]
