import pandas as pd
import numpy as np
import schedule
import time
import re
import nltk
import pymysql  # Import pymysql untuk SQLAlchemy
from memory_profiler import memory_usage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sqlalchemy import create_engine
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import cross_val_score
from threading import Thread

# Download NLTK resources
nltk.download('stopwords', download_dir='/usr/local/nltk_data')
nltk.download('punkt', download_dir='/usr/local/nltk_data')
nltk.download('wordnet', download_dir='/usr/local/nltk_data')

# Tambahkan lokasi data NLTK agar bisa ditemukan
nltk.data.path.append('/usr/local/nltk_data')

# Database configuration
DATABASE_URI = 'mysql+pymysql://root:nudgIcUzPEjPJwiBqpopSgkYSDUTsnuX@maglev.proxy.rlwy.net:14974/railway?charset=utf8mb4'

# Inisialisasi stopwords dan lemmatizer
try:
    stop_words = stopwords.words('indonesian')
except:
    nltk.download('stopwords')
    stop_words = stopwords.words('indonesian')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
custom_stopwords = ['mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali', 
                    'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
all_stop_words = stop_words + custom_stopwords

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Fungsi untuk melakukan stemming dan lemmatization
def preprocess_text(text):
    text = clean_text(text)
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

# Fungsi untuk mengambil data dari database dan memperbarui dataframe
def fetch_data():
    global df, model
    try:
        engine = create_engine(DATABASE_URI)
        with engine.connect() as connection:
            query = "SELECT description, component FROM training_data"
            df = pd.read_sql(query, connection)
        
        if df.empty:
            print("Database kosong, tidak ada data untuk melatih model.")
            return

        df.drop_duplicates(inplace=True)
        df['description'] = df['description'].str.lower().apply(preprocess_text)

        # Training ulang model
        print("Data dan model diperbarui dari database.")
        X = df['description']
        y = df['component']
        tfidf = TfidfVectorizer(stop_words=all_stop_words)
        model = make_pipeline(tfidf, MultinomialNB(alpha=1.0))
        model.fit(X, y)

        # Perform 5-fold cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        print("Accuracy for each fold:", scores)
        print(f"Mean accuracy: {scores.mean():.2f}")
    except Exception as e:
        print(f"Error saat mengambil data dari database: {str(e)}")

# Jalankan update pertama kali
fetch_data()

# Fungsi prediksi
def predict_faulty_component(description):
    description = preprocess_text(description)
    try:
        probas = model.predict_proba([description])[0]
        components = model.classes_
        predictions = {components[i]: np.round(probas[i] * 100, 2) for i in range(len(components))}
        return dict(sorted(predictions.items(), key=lambda item: item[1]))
    except Exception as e:
        return {"error": f"Model belum siap atau error: {str(e)}"}

# Buat Flask app untuk menerima request prediksi
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        description = data.get('description', '')
        if not description:
            return jsonify({'error': 'Deskripsi tidak boleh kosong'}), 400

        predictions = predict_faulty_component(description)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Fungsi untuk menjalankan scheduler
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(10)  # Cek setiap 10 detik

# Jalankan Flask server dan scheduler
if __name__ == '__main__':
    # Jalankan Flask di thread lain
    flask_thread = Thread(target=lambda: app.run(debug=True, use_reloader=False, host="0.0.0.0"))
    flask_thread.start()

    # Jalankan scheduler di thread terpisah agar tidak mengganggu Flask
    scheduler_thread = Thread(target=run_scheduler)
    scheduler_thread.start()
