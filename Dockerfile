FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y gcc curl unzip libmysqlclient-dev python3-dev

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Setup virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data during build step
RUN mkdir -p /usr/local/nltk_data \
 && python -m nltk.downloader -d /usr/local/nltk_data stopwords punkt wordnet

# Set env so NLTK knows where to find data
ENV NLTK_DATA="/usr/local/nltk_data"

# Jalankan aplikasi
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
