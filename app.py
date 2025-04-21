import os
import re
import time
import nltk
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

from sqlalchemy import create_engine
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Konfigurasi NLTK
nltk.data.path.append('/usr/local/nltk_data')

# Database config (Railway friendly)
DATABASE_URI = os.getenv('DATABASE_URL', 'mysql+pymysql://root:@localhost/compere_tugasakhir')
engine = create_engine(DATABASE_URI)

# Stopwords & Preprocess tools
stop_words = stopwords.words('indonesian')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
custom_stopwords = ['mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali', 'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
all_stop_words = stop_words + custom_stopwords

def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'\s+', ' ', text)  # Hapus spasi berlebih
    text = re.sub(r'[^\w\s]', '', text)  # Hapus karakter spesial
    return text

def preprocess_text(text):
    text = clean_text(text)
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

# Global model
model = None

# Training Function
def fetch_and_train_model():
    global model
    start_time = time.time()

    query = "SELECT description, component FROM training_data"
    df = pd.read_sql(query, engine)
    df.drop_duplicates(inplace=True)
    df['description'] = df['description'].apply(preprocess_text)

    X = df['description']
    y = df['component']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = make_pipeline(TfidfVectorizer(stop_words=all_stop_words), MultinomialNB(alpha=1.0))
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean()}")

    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Waktu training: {time.time() - start_time:.2f} detik")

def predict_faulty_component(description):
    if model is None:
        raise Exception("Model belum di-train. Silakan akses /train terlebih dahulu.")
    description = preprocess_text(description)
    probas = model.predict_proba([description])[0]
    components = model.classes_
    predictions = {components[i]: np.round(probas[i] * 100, 2) for i in range(len(components))}
    return dict(sorted(predictions.items(), key=lambda item: item[1]))

# Flask App
app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    try:
        fetch_and_train_model()
        return jsonify({'status': 'Model trained successfully! ✅'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        description = data['description']
        predictions = predict_faulty_component(description)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Jalankan Server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)