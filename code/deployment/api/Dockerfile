FROM python:3.11.9-slim

WORKDIR /api

COPY model_dir/ .
RUN pip install -r requirements.txt

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model_api.py .

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "6000"]
