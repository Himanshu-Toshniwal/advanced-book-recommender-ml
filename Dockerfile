FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Matplotlib non-interactive backend (no display needed in container)
ENV MPLBACKEND=Agg

# System deps for numpy/sklearn + matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-generate sample data so container works without Kaggle credentials
RUN echo "n" | python src/download_data.py

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pandas, sklearn, matplotlib; print('ok')"

CMD ["python", "src/main.py"]
