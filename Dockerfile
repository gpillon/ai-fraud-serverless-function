FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY scaler.pkl .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
