# CUDA 12.4 Base-Image (cuDNN nicht als Tag verfügbar)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3 default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pget for fast downloads (used by RunPod)
RUN wget https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64 -O /usr/local/bin/pget && \
    chmod +x /usr/local/bin/pget

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Run your app
CMD ["python", "predict.py"]
