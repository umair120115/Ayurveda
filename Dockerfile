# === Stage 1: Build Stage ===
# Use a full Python image to build dependencies
FROM python:3.12 as builder

WORKDIR /app

RUN pip install --upgrade pip

# Copy the requirements file from your project root
COPY requirements.txt .

# Install dependencies (and gunicorn for production)
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# === Stage 2: Final Stage ===
# Use a minimal, secure "slim" image for the final product
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for the Doppler CLI
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the Doppler CLI
RUN curl -Ls --tlsv1.2 --proto "=https" --max-time 10 https://install.doppler.com/production/cli/install.sh | sh

# Create a non-root user for security
RUN addgroup --system app && adduser --system --group app

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# --- THIS IS THE KEY CHANGE ---
# Copy *only* the 'app' directory contents into the container's /app directory
# This copies Ayurveda/app/main.py -> /app/main.py
COPY app/ .

# Change ownership of the app directory to the non-root user
RUN chown -R app:app /app

# Switch to the non-root user
USER app

# --- Google Cloud Run Specifics ---
# Cloud Run automatically provides a $PORT environment variable.
# We set a default (8080), but Cloud Run will override this.
ENV PORT 8080

EXPOSE 8080

# --- The Final Command ---
# This command runs from the /app directory,
# so gunicorn will find main:app (from your main.py file)
CMD ["doppler", "run", "--", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT"]