FROM python:3.12.1

WORKDIR /app

# Copy dan install dependensi langsung tanpa venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]