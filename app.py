import pandas as pd
import numpy as np
import time
import re
import nltk
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

# Jangan gunakan /usr/local/nltk_data, gunakan /app/nltk_data
nltk.download('stopwords', download_dir='/app/nltk_data')
nltk.download('punkt', download_dir='/app/nltk_data')
nltk.download('wordnet', download_dir='/app/nltk_data')

# Pastikan lokasi ini dikenali NLTK
nltk.data.path.append('/app/nltk_data')

# Database configuration
DATABASE_URI = 'mysql+pymysql://root:nudgIcUzPEjPJwiBqpopSgkYSDUTsnuX@maglev.proxy.rlwy.net:14974/railway?charset=utf8mb4'
engine = create_engine(DATABASE_URI)

# Kamus sinonim atau variasi frasa
synonym_dict = {
    'lag': 'performa lambat',
    'tidak mau hidup': 'tidak menyala',
    'tidak berfungsi': 'tidak bekerja',
    'mati mendadak': 'tiba-tiba mati',
    'sering restart': 'sering dimulai ulang'
}

start_time = time.time()

# Inisialisasi stopwords dan lemmatizer
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
    
    # Ganti kata atau frasa sesuai kamus sinonim
    #for key, value in synonym_dict.items():
    #    text = text.replace(key, value)
    
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

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

        # Perbarui jumlah baris terakhir
        last_row_count = current_row_count

# Jalankan update pertama kali
fetch_data()

# End time for measuring the initial data fetch and training
end_time = time.time()
print("Waktu eksekusi awal:", end_time - start_time, "detik")

# Fungsi prediksi
def predict_faulty_component(description):
    description = preprocess_text(description)
    probas = model.predict_proba([description])[0]
    components = model.classes_
    predictions = {components[i]: np.round(probas[i] * 100, 2) for i in range(len(components))}
    return dict(sorted(predictions.items(), key=lambda item: item[1]))

# Buat Flask app untuk menerima request prediksi
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

# Jalankan Flask server 
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Gunakan PORT dari Railway, default ke 5000 jika tidak ada
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)