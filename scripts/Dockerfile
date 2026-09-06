# Dockerfile for Ustad - Local LLM Distillation Studio
#
# This creates a containerized version of Ustad with all dependencies.
# Note: GPU support requires nvidia-docker and proper driver setup on the host.
#
# Build:
#   docker build -t ustad:latest -f scripts/Dockerfile .
#
# Run (CPU only):
#   docker run -p 8177:8177 -v $(pwd)/data:/app/data ustad:latest
#
# Run (with GPU):
#   docker run --gpus all -p 8177:8177 -v $(pwd)/data:/app/data ustad:latest

FROM python:3.12-slim

LABEL maintainer="Ustad Team"
LABEL description="Local-first LLM distillation studio"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install PyTorch (CUDA 12.6 version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8177

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8177/api/system || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/data/hf-cache

# Run the server
CMD ["python", "-m", "backend.server"]
