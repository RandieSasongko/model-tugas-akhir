FROM python:3.11

WORKDIR /app

# Copy dan install dependensi langsung tanpa venv
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]