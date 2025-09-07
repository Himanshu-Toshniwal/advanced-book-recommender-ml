FROM python:3.10-slim         
WORKDIR /app                  
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1        

# System dependencies
RUN apt-get update && apt-get install -y gcc g++  

# Python dependencies  
COPY requirements.txt .        # ✅ Copy requirements first
RUN pip install --no-cache-dir -r requirements.txt  # ✅ Install deps

# Application code
COPY . .                       # ✅ Copy all files

# Security
RUN useradd --create-home --shell /bin/bash app  # ✅ Non-root user
USER app                       # ✅ Switch to app user

# Runtime
EXPOSE 8000                    # ✅ Port exposure
HEALTHCHECK ...                # ✅ Health monitoring
CMD ["python", "src/main.py"]  # ✅ Start application
