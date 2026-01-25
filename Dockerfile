# Parking Spot Monitoring Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLOv8 model at build time (without importing cv2)
RUN python -c "from ultralytics.utils.downloads import download; download('https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt', dir='.')"

# Copy application code
COPY src/ ./src/

# Copy config files
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 9878

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9878/api/v1/health')" || exit 1

# Run the application
CMD ["python", "-m", "parking_monitor.main"]
