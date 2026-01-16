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
    requests

# Copy the entire project
COPY . .

# Make entrypoint executable
RUN chmod +x /app/docker/entrypoint.sh

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SIMULATOR_PORT=8080
ENV WORKER_ID=0

# Expose simulator port (internal to container)
EXPOSE 8080

# Use the worker entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]
