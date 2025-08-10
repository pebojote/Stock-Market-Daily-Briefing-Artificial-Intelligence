# Stage 1: Build the application environment
# Use a Python base image with a smaller footprint
FROM python:3.11-slim as builder

# Set the working directory
WORKDIR /app

# Install dependencies needed for packaging
RUN pip install --no-cache-dir poetry

# Copy only the files needed to install dependencies
COPY poetry.lock pyproject.toml ./

# Install project dependencies
RUN poetry install --no-root --no-dev --no-interaction --no-ansi

# Stage 2: Create the final runtime image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY main.py .

# Expose the port the Flask app runs on
EXPOSE 8080

# Define the command to run the application
# Use gunicorn or another production-ready server for robustness
CMD ["poetry", "run", "python", "main.py"]