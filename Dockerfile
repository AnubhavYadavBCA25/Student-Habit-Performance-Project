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

# Expose the port Flask is running on
# EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]