# Dockerfile for running infer.py server on port 80

FROM python:3.11

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 80
EXPOSE 80

# Run server
CMD ["python", "infer.py"]