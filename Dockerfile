# Use the official PyTorch base image to ensure CUDA compatibility and avoid image resolution errors.
# This image comes with PyTorch, CUDA 12.1, and cuDNN pre-installed.
FROM nvcr.io/nvidia/pytorch:24.05-py3


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies that might be missing from the base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pget for fast downloads
RUN wget https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64 -O /usr/local/bin/pget && \
    chmod +x /usr/local/bin/pget

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
# PyTorch is already included in the base image, so we just install the other packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Command to run the worker
CMD ["python", "predict.py"] 