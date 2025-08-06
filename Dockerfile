# Multi-stage Docker build for REBALANCE toolkit
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && \
    pip install .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r rebalance && useradd -r -g rebalance rebalance

# Create working directory
WORKDIR /app

# Copy application files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

# Set ownership
RUN chown -R rebalance:rebalance /app
USER rebalance

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.bias_detection.detector; print('OK')" || exit 1

# Default command
CMD ["rebalance", "--help"]

# Development stage (optional)
FROM production as development

USER root

# Install development dependencies
RUN pip install -e ".[dev,external]"

# Install additional tools for development
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER rebalance

# Expose Jupyter port for development
EXPOSE 8888

CMD ["bash"]