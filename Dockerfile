# Gunakan base image yang lebih ringan
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy semua file ke dalam container
COPY . /app

# Buat virtual environment
RUN python -m venv /opt/venv

# Aktifkan venv dan install dependencies
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install -r requirements.txt

# Pastikan virtual environment digunakan di setiap run
ENV PATH="/opt/venv/bin:$PATH"