FROM python:3.10.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY monitoring/ monitoring/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV STREAMLIT_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose ports for both applications
EXPOSE ${PORT}
EXPOSE ${STREAMLIT_PORT}

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"] 