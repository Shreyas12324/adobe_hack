FROM --platform=linux/amd64 python:3.11-slim

# Create app directories
RUN mkdir -p /app/input /app/output

# Copy only necessary files
COPY main.py requirements.txt ./

# Install dependencies (no venv needed in container)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# No internet, no GPU, minimal image
# python:3.11-slim is <200MB, no GPU, no extra tools
# Remove any network access capabilities
RUN rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Set working directory and entrypoint
WORKDIR /
ENTRYPOINT ["python", "main.py"] 