# Use an official lightweight Python image as base
FROM python:3.9-slim AS base

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install necessary Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port (Flask default is 5000)
EXPOSE 5000

# Development stage
FROM base AS development
CMD ["python", "src/api.py"]

# Production stage with Uvicorn
FROM base AS production
RUN pip install --no-cache-dir uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "5000"]
