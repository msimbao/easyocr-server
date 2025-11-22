FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models (this is the key part!)
# This runs during build, so the model is baked into the image
RUN python -c "import easyocr; reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)"

# Copy application code
COPY main.py .

# Expose port (Render will use PORT env variable)
EXPOSE 10000

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}