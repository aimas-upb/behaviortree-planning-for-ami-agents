# Dockerfile for BT Planning Experiment Workers
# Each container runs its own simulator + processes experiments from a work queue

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for experiments
RUN pip install --no-cache-dir \
    openai \
    python-dotenv \
    pandas \
    matplotlib \
    pyyaml \
    requests \
    fastembed \
    tqdm \
    rich

# Pre-download fastembed model so it's available at runtime (even as non-root)
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
    && chmod -R a+rX /app/.fastembed_cache

# Copy the entire project
COPY . .

# Make entrypoints executable
RUN chmod +x /app/docker/entrypoint.sh /app/docker/entrypoint_experience.sh

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SIMULATOR_PORT=8080
ENV WORKER_ID=0

# Expose simulator port (internal to container)
EXPOSE 8080

# Use the worker entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]
