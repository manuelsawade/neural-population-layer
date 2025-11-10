# Use Python 3.9 with CUDA support for GPU acceleration
FROM python:3.12.12-slim

ENV PYTHONUNBUFFERED=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Create results and data directories (data will be mounted at runtime)
RUN mkdir -p results data

# Set Python path to include src directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Set the working directory to src for script execution
WORKDIR /app/src

# Default command to run the tuner script
CMD ["python", "tuner_mnist_concept.py"]