FROM python:3.11

WORKDIR /app

# Copy dan install dependensi langsung tanpa venv
COPY requirements.txt .

COPY . .

CMD ["python", "app.py"]