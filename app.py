import pandas as pd
import numpy as np
import time
import re
import nltk
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sqlalchemy import create_engine
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score

# Setup NLTK
nltk.download('stopwords', download_dir='/usr/local/nltk_data')
nltk.download('punkt', download_dir='/usr/local/nltk_data')
nltk.download('wordnet', download_dir='/usr/local/nltk_data')
nltk.data.path.append('/usr/local/nltk_data')

# Config
DATABASE_URI = 'mysql+pymysql://root:GcchhrdqnsKyauycgVpnYKpXMzYSELhn@ballast.proxy.rlwy.net:58414/railway?charset=utf8mb4'
MODEL_PATH = 'model.joblib'

# Init Flask app
app = Flask(__name__)

# DB Engine
engine = create_engine(DATABASE_URI)

# Stopwords & preprocessing tools
stop_words = stopwords.words('indonesian')
custom_stopwords = ['mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali', 
                    'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
all_stop_words = stop_words + custom_stopwords
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Clean text function
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Preprocess function
def preprocess_text(text):
    text = clean_text(text.lower())
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

# Training model
def fetch_and_train_model():
    global model
    query = "SELECT description, component FROM training_data"
    df = pd.read_sql(query, engine)
    df.drop_duplicates(inplace=True)
    df['description'] = df['description'].apply(preprocess_text)

    X = df['description']
    y = df['component']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = make_pipeline(TfidfVectorizer(stop_words=all_stop_words), MultinomialNB(alpha=1.0))
    
    # Cross-validation (optional, for debugging/tracking performance)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())

    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)
    print("Test accuracy:", test_accuracy)

    # Save model
    joblib.dump(model, MODEL_PATH)

# Load or train model
start_time = time.time()
if os.path.exists(MODEL_PATH):
    print("Loading model dari file...")
    model = joblib.load(MODEL_PATH)
else:
    print("Training model karena file tidak ditemukan...")
    fetch_and_train_model()
end_time = time.time()
print("Waktu inisialisasi:", end_time - start_time, "detik")

# Predict function
def predict_faulty_component(description):
    description = preprocess_text(description)
    probas = model.predict_proba([description])[0]
    components = model.classes_
    predictions = {components[i]: np.round(probas[i] * 100, 2) for i in range(len(components))}
    return dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        description = data['description']
        predictions = predict_faulty_component(description)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Optional: Endpoint untuk retraining model manual
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        fetch_and_train_model()
        return jsonify({'message': 'Model retrained successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
