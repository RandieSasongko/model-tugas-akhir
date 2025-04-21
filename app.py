import os
import re
import nltk
import joblib
import numpy as np
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Setup NLTK path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Load NLTK data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# Inisialisasi tools NLP
stop_words = stopwords.words('indonesian')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
custom_stopwords = ['mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali', 
                    'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
all_stop_words = stop_words + custom_stopwords

# Text cleaning
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Preprocessing
def preprocess_text(text):
    text = clean_text(text)
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

# Load model
print("🚀 Loading model...")
model = joblib.load("model.joblib")
print("✅ Model loaded!")

# Flask setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        description = data['description']
        cleaned = preprocess_text(description)
        probas = model.predict_proba([cleaned])[0]
        components = model.classes_
        predictions = {components[i]: round(probas[i]*100, 2) for i in range(len(components))}
        return jsonify(dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True)))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Start server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
