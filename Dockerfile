FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-download all Hugging Face model weights for offline execution
# This ensures the container can run inference without internet access.
# Models cached: Whisper-tiny (ASR), Wav2Vec2-ER (SER), DistilBERT (NLP)
RUN python scripts/cache_models.py

# Default entry point: run the full multimodal pipeline
CMD ["python", "main.py", "--config", "configs/config.yaml"]
