FROM python:3.11-slim


COPY requirements.txt .
RUN pip install uv
RUN uv pip install -r requirements.txt --system


COPY src src
COPY .env .env


CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
