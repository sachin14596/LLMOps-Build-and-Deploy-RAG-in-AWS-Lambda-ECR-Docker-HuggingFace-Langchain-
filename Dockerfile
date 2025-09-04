FROM python:3.13-slim-bookworm

# Good defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install AWS CLI v2 at OS level (no pip) to avoid botocore conflicts
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl unzip \
 && rm -rf /var/lib/apt/lists/* \
 && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
 && unzip awscliv2.zip \
 && ./aws/install \
 && rm -rf awscliv2.zip aws

# Copy dependency files first (better build caching)
COPY requirements.txt constraints.txt ./

# Upgrade pip and install Python deps using constraints (faster, deterministic)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy the rest of your application code
COPY . .

# Expose Streamlit’s default port and run the app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
