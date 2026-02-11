FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

COPY . .

# Generate demo data if not present
RUN python -c "from ml.pipeline.run_all import run_pipeline; run_pipeline()" 2>/dev/null || true

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
