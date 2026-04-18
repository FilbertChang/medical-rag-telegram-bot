# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Default command (can be overridden by docker-compose)
CMD ["python", "bot.py"]