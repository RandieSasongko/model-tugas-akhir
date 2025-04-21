from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

print("🟢 Starting app.py")

# Set path lokal untuk nltk
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Pastikan semua resource sudah tersedia di folder nltk_data
# Jangan download saat runtime (hindari nltk.download(...))

# Load stopwords, tokenizer, lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

print("📦 NLTK components loaded")

# Load model
model = joblib.load("model.joblib")
print("✅ Model loaded.")

# Init Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def index():
    return "🚀 Model API is running!"

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    print(f"📨 Received text: {text}")

    # Basic preprocessing
    tokens = word_tokenize(text.lower())
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    processed_text = " ".join(filtered)

    print(f"🧹 Processed text: {processed_text}")

    # Vectorize or transform (contoh, tergantung model kamu)
    # X = vectorizer.transform([processed_text])  # Uncomment jika pakai vectorizer
    prediction = model.predict([processed_text])  # Atur sesuai model kamu

    print(f"📊 Prediction result: {prediction[0]}")

    return jsonify({"prediction": prediction[0]})
