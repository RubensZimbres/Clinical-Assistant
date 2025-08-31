FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 8080

ENV PORT=8080
ENV HOST=0.0.0.0

CMD ["python3 uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
