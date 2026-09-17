FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: PyMuPDF's runtime needs these even headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY evals ./evals
COPY ml_artifacts ./ml_artifacts
COPY examples ./examples

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
