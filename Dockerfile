# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt if available
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Explicitly copy artifacts folder (optional if above COPY . . already includes it)
COPY artifacts/ ./artifacts/

# Run the Flask app
CMD ["python", "app.py"]