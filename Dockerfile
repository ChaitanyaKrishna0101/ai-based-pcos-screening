FROM python:3.11-slim

WORKDIR /code

# System deps needed by xgboost/shap wheels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at build time so the image is ready to serve immediately.
# (Uses the bundled synthetic dataset unless data/pcos_dataset.csv is replaced
#  with the real Kaggle CSV before building.)
RUN python app/model/train.py || true

# Hugging Face Spaces expects the app on port 7860
ENV PORT=7860
EXPOSE 7860

# Non-root user (HF Spaces best practice)
RUN useradd -m appuser && chown -R appuser /code
USER appuser

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app.main:app"]
