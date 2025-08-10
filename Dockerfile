# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install dependencies, no need for Gunicorn
RUN pip install --no-cache-dir python-dotenv flask openai markdown

# Copy the rest of the application code
COPY . .

# Run the application directly.
# The script will execute the daily_job() and then exit.
CMD ["python3", "main.py"]
