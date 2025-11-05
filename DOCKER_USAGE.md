# Docker Usage Instructions

This project includes Docker configurations to run the neural population layer experiments in a containerized environment with optimized data handling.

## Files Created

- `Dockerfile` - Standard CPU-based container (Python 3.12.12)
- `Dockerfile.gpu` - GPU-enabled container with CUDA support
- `docker-compose.yml` - Easy orchestration with optimized volume mounts
- `.dockerignore` - Optimized build context excluding data files

## Quick Start

### CPU-only Version

```bash
# Build and run with Docker Compose (recommended)
docker-compose up neural-population-layer

# Or build and run manually
docker build -t neural-population-layer .
docker run -v "$(pwd)/results:/app/results" -v "$(pwd)/data:/app/data" neural-population-layer
```

### GPU Version (Recommended for faster training)

```bash
# Build and run GPU version with Docker Compose
docker-compose up neural-population-layer-gpu

# Or build and run manually
docker build -f Dockerfile.gpu -t neural-population-layer-gpu .
docker run --gpus all -v "$(pwd)/results:/app/results" -v "$(pwd)/data:/app/data" neural-population-layer-gpu
```

## Prerequisites

- Docker installed
- For GPU version: NVIDIA Docker runtime (`nvidia-docker2`)
- **Local CIFAR-10 dataset** in `./data/cifar-10-batches-py/` (see Data Setup below)

## Data Setup

**Important**: To avoid re-downloading the CIFAR-10 dataset every time:

1. **If you already have the dataset**: Ensure it's in `./data/cifar-10-batches-py/` directory
2. **If you don't have the dataset**:
   - Run the script once locally to download: `python src/tuner_cifar10.py` (will download to `./data/`)
   - Or temporarily comment out the data volume mount in docker-compose.yml for the first run

The current setup mounts your local `./data` directory to avoid redundant downloads.

## Volume Mounts

The containers mount the following directories:

- `./results` - Experiment results will be saved here and persisted on the host
- `./data` - Dataset storage (mounts existing local datasets to avoid re-downloading)

## Performance Optimizations

- **Shared Memory**: Configured with 10GB shared memory (`shm_size: '10gb'`) for Ray performance
- **Data Persistence**: Local datasets are mounted to avoid re-downloading
- **Caching**: Docker layer caching optimized by copying requirements first
- **Ray Configuration**: Environment variables set to prevent redundant downloads

## Customization

To run with different parameters or scripts:

```bash
# Run a different script
docker-compose run neural-population-layer python main_mnist.py

# Interactive mode for development
docker-compose run neural-population-layer bash

# Override number of samples
docker-compose run neural-population-layer python -c "
import sys; sys.path.append('/app/src')
from tuner_cifar10 import main
# Modify the config in main() function or run with custom parameters
"
```

## Troubleshooting

### Dataset Issues

- **"FileNotFoundError" for CIFAR-10**: Ensure `./data/cifar-10-batches-py/` exists locally
- **Slow downloads**: The container now uses mounted data to avoid re-downloading

### Memory Issues

- **Ray warnings about /dev/shm**: The docker-compose.yml includes `shm_size: '10gb'` to fix this
- **Out of memory**: Reduce `num_samples` or `max_concurrent_trials` in the script

### GPU Issues

- **CUDA not found**: Ensure nvidia-docker is installed and GPU service is used
- **GPU memory**: Monitor GPU memory usage with `nvidia-smi`

## Performance Notes

- The GPU version (`Dockerfile.gpu`) is recommended for faster training
- Ray Tune will automatically utilize available resources
- Results are persisted in the `./results` directory on your host machine
- Dataset mounting eliminates download time and reduces container startup time
