# 1. Base image: Python runtime
FROM python:3.10-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy project files into container
COPY pyproject.toml .
COPY src/ src/

# 4. Install dependencies
RUN pip install --upgrade pip
RUN pip install -e .

# 5. Expose API port
EXPOSE 8000

# 6. Run FastAPI server
CMD ["uvicorn", "credit_risk_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
