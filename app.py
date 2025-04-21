import os
import re
import joblib
import nltk
import numpy as np
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load NLTK data
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Load stopwords dan inisialisasi NLP tools
stop_words = stopwords.words('indonesian')
custom_stopwords = ['mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali', 
                    'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
all_stop_words = stop_words + custom_stopwords

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Fungsi preprocessing
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_text(text):
    text = clean_text(text)
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

# Load model yang sudah ditraining
MODEL_PATH = 'model.joblib'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' tidak ditemukan.")
model = joblib.load(MODEL_PATH)

# Fungsi prediksi
def predict_faulty_component(description):
    description = preprocess_text(description)
    probas = model.predict_proba([description])[0]
    components = model.classes_
    predictions = {components[i]: round(probas[i] * 100, 2) for i in range(len(components))}
    return dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

# Inisialisasi Flask app
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

# Jalankan server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
