FROM python:3.9-slim-buster

WORKDIR /app

# 1) Copy dependency files first (better build caching)
COPY requirements.txt constraints.txt ./

# 2) Upgrade pip and install Python deps using constraints
#    This avoids slow resolver backtracking during Docker builds
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# 3) Copy the rest of your application code
COPY . .

# 4) Expose Streamlit’s default port and run the app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
