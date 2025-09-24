# Dockerfile for Mitochondria Analyzer
# Optimized for cross-platform deployment

FROM python:3.10-slim

# Install system dependencies needed for OpenCV and GUI libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt /app/requirements.txt
COPY MoDL/requirements.txt /app/modl-requirements.txt

# Install Python dependencies in correct order
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy>=1.21.0,<1.25.0" && \
    pip install --no-cache-dir streamlit pandas && \
    pip install --no-cache-dir -r modl-requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p data/raw data/segmented data/analyzed data/jobs logs .streamlit

# Create Streamlit config
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
maxUploadSize = 1000\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "light"' > .streamlit/config.toml

# Set environment variables
ENV PYTHONPATH="/app"
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# TensorFlow memory optimization - prevent large chunk allocation
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_MEMORY_FRACTION=0.4

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]