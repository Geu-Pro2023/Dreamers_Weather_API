# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 10000

# Run the app
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=10000"]
