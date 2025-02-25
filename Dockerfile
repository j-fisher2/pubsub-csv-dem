# Use Python 3.9 slim base image for efficiency
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements3.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements3.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy script and any local files
COPY . .

# Command for Dataflow worker
CMD ["python", "pipeline.py"]