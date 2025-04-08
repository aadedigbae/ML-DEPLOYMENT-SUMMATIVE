# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Avoid pywin32 which causes issues in Linux containers
RUN grep -v "pywin32" requirements.txt > temp_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r temp_requirements.txt && rm temp_requirements.txt

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "src/api.py"]
