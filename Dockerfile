FROM python:3.9-slim
WORKDIR /app

# Copy the source code into the image
COPY . /app

# Update package lists, install awscli, and remove package lists to reduce image size
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies without caching to keep the image lean
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run your application (adjust as needed)
CMD ["python", "frontend.py"]
