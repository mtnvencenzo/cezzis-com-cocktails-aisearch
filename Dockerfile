# Multi-stage build - use ubuntu with pre-installed tools for speed
FROM python:3.12-bullseye AS builder

# Setup poetry for build and dependency management
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create true
RUN poetry config installer.parallel true

# Set working directory
WORKDIR /app/src

# Copy only dependency files first (for better caching)
COPY ./pyproject.toml ./poetry.lock ./README.md ./
COPY ./src/cezzis_com_cocktails_aisearch/ ./cezzis_com_cocktails_aisearch/

# Install dependencies with caching optimizations
RUN poetry install --only=main

# Build the application
RUN poetry build -o dist -v

# Final stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy and install the built package efficiently
COPY --from=builder /app/src/dist/*.tar.gz ./
RUN pip install --no-cache-dir --disable-pip-version-check *.tar.gz && rm *.tar.gz
COPY ./src/cezzis_com_cocktails_aisearch/static ./static

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

EXPOSE 8010

# Run the application using the installed package
CMD ["uvicorn", "cezzis_com_cocktails_aisearch.main:app", "--host", "0.0.0.0", "--port", "8010"]