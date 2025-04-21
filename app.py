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
from sklearn.model_selection import train_test_split, cross_val_score
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sqlalchemy import create_engine
import joblib

# Download NLTK resources
nltk.download('stopwords', download_dir='/nltk_data')
nltk.download('punkt', download_dir='/nltk_data')
nltk.download('wordnet', download_dir='/nltk_data')

# Tambahkan lokasi data NLTK agar bisa ditemukan
nltk.data.path.append('/nltk_data')

# Database configuration
DATABASE_URI = 'mysql+pymysql://root:GcchhrdqnsKyauycgVpnYKpXMzYSELhn@ballast.proxy.rlwy.net:58414/railway?charset=utf8mb4'
engine = create_engine(DATABASE_URI)

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
    
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

# Inisialisasi variabel untuk menghitung baris
last_row_count = 0

# Fungsi untuk fetch dan latih model
def fetch_and_train_model():
    global model, last_row_count
    query = "SELECT description, component FROM training_data"
    df = pd.read_sql(query, engine)
    df.drop_duplicates(inplace=True)
    df['description'] = df['description'].str.lower().apply(preprocess_text)
    
    X = df['description']
    y = df['component']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Pipeline dan cross-validation
    model = make_pipeline(TfidfVectorizer(stop_words=all_stop_words), MultinomialNB(alpha=1.0))

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())

    # Train final model
    model.fit(X_train, y_train)

    # Evaluate on test set
    test_accuracy = model.score(X_test, y_test)
    print("Test accuracy:", test_accuracy)

    # Simpan model ke disk
    joblib.dump(model, 'model.pkl')

# Cek apakah model sudah ada di disk
def load_model():
    if os.path.exists('model.pkl'):
        return joblib.load('model.pkl')
    else:
        fetch_and_train_model()
        return joblib.load('model.pkl')

# Load model saat aplikasi pertama kali berjalan
model = load_model()

# End time untuk mengukur waktu eksekusi awal
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
