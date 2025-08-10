FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# This is the crucial part: tell the container to run the script
CMD ["python3", "main.py"]
