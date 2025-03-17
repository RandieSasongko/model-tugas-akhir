# Gunakan base image Python
FROM python:3.12

# Atur working directory di dalam container
WORKDIR /app

# Copy semua file ke dalam container
COPY . .

# Install semua dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Download dataset NLTK agar tidak hilang saat restart
RUN python -m nltk.downloader -d /usr/local/nltk_data punkt stopwords wordnet

# Expose port 8080 untuk Flask
EXPOSE 8080

# Jalankan aplikasi Flask dengan Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
