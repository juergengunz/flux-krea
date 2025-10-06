# CUDA 12.4 Base-Image with cuDNN for better tensor operations
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables for optimal GPU and memory usage
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

# Install system dependencies and Python with additional libraries for better tensor handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install pget for fast downloads (used by RunPod)
RUN wget https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64 -O /usr/local/bin/pget && \
    chmod +x /usr/local/bin/pget

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install Python dependencies (using standard PyPI builds to match Cog)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .


# Download SRPO model during build time to avoid downloading at runtime
RUN mkdir -p /app/srpo && \
    pget https://huggingface.co/tencent/SRPO/resolve/main/diffusion_pytorch_model.safetensors \
    /app/srpo/diffusion_pytorch_model.safetensors

# Set Python path and additional environment variables for better GPU memory management
ENV PYTHONPATH=/app
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_LAUNCH_BLOCKING=0

# Run your app
CMD ["python", "predict.py"]
