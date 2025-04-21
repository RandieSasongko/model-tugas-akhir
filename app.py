import pandas as pd
import numpy as np
import time
import re
import nltk
import os
from memory_profiler import memory_usage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sqlalchemy import create_engine
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK resources
nltk.download('stopwords', download_dir='/usr/local/nltk_data')
nltk.download('punkt', download_dir='/usr/local/nltk_data')

# Tambahkan lokasi data NLTK agar bisa ditemukan
nltk.data.path.append('/usr/local/nltk_data')

# Database configuration
DATABASE_URI = 'mysql+pymysql://root:GcchhrdqnsKyauycgVpnYKpXMzYSELhn@ballast.proxy.rlwy.net:58414/railway?charset=utf8mb4'
engine = create_engine(DATABASE_URI)

start_time = time.time()

# Inisialisasi stopwords dan stemmer
stop_words = stopwords.words('indonesian')
custom_stopwords = ['mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali', 
                    'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
all_stop_words = stop_words + custom_stopwords

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()  # lowercase semua
    text = re.sub(r'\d+', '', text)  # hapus angka
    text = re.sub(r'\s+', ' ', text)  # hapus spasi berlebih
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    return text.strip()

# Fungsi untuk preprocess (bersihkan, ganti sinonim, stopwords, stemming)
def preprocess_text(text):
    text = clean_text(text)

    words = word_tokenize(text)  # tokenisasi
    words = [word for word in words if word not in all_stop_words]  # hapus stopwords
    text = ' '.join(words)

    # Stemming dengan Sastrawi
    stemmed_text = stemmer.stem(text)
    return stemmed_text

# Inisialisasi variabel untuk menghitung baris
last_row_count = 0

# Fungsi untuk mengambil data dari database dan memperbarui dataframe
def fetch_data():
    global df, model, last_row_count
    query_count = "SELECT COUNT(*) FROM training_data"
    current_row_count = pd.read_sql(query_count, engine).iloc[0, 0]
    
    # Hanya memperbarui jika ada tambahan 100 data atau lebih
    if current_row_count - last_row_count >= 100:
        query = "SELECT description, component FROM training_data"
        df = pd.read_sql(query, engine)
        df.drop_duplicates(inplace=True)
        df['description'] = df['description'].apply(preprocess_text)
        
        # Training ulang model
        print("Data dan model diperbarui dari database.")
        X = df['description']
        y = df['component']
        tfidf = TfidfVectorizer(stop_words=all_stop_words)
        model = make_pipeline(tfidf, MultinomialNB(alpha=1.0))
        model.fit(X, y)

        # Cross-validation 5-fold
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        print("Accuracy for each fold:", scores)
        print(f"Mean accuracy: {scores.mean():.2f}")

        last_row_count = current_row_count

# Jalankan update pertama kali
fetch_data()

end_time = time.time()
print("Waktu eksekusi awal:", end_time - start_time, "detik")

# Fungsi prediksi
def predict_faulty_component(description):
    description = preprocess_text(description)
    probas = model.predict_proba([description])[0]
    components = model.classes_
    predictions = {components[i]: np.round(probas[i] * 100, 2) for i in range(len(components))}
    return dict(sorted(predictions.items(), key=lambda item: item[1]))

# Buat Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        description = data['description']
        predictions = predict_faulty_component(description)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Jalankan server Flask
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Railway PORT env
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
