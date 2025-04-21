import os
import re
import numpy as np
from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from joblib import load

# Set path ke data NLTK (pastikan folder ini ada dalam image Docker)
nltk.data.path.append('/usr/local/nltk_data')

# Inisialisasi NLTK tools
stop_words = stopwords.words('indonesian')
custom_stopwords = [
    'mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali',
    'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya'
]
all_stop_words = stop_words + custom_stopwords
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_text(text):
    text = clean_text(text.lower())
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

# Load model yang sudah dilatih
model = load('model.joblib')

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
        if not data or 'description' not in data:
            return jsonify({'error': 'Deskripsi tidak ditemukan dalam request'}), 400
        
        description = data['description']
        predictions = predict_faulty_component(description)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Jalankan server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
