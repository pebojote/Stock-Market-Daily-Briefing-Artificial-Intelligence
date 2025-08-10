# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install dependencies, including Gunicorn for production
RUN pip install --no-cache-dir gunicorn python-dotenv flask openai markdown

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Run the application with Gunicorn
# Gunicorn will listen on the PORT environment variable automatically provided by Cloud Run.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
