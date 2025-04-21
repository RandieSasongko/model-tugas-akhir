import pandas as pd
import numpy as np
import re
import os
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Database connection
DATABASE_URI = 'mysql+pymysql://root:GcchhrdqnsKyauycgVpnYKpXMzYSELhn@ballast.proxy.rlwy.net:58414/railway?charset=utf8mb4'
engine = create_engine(DATABASE_URI)

# NLTK preparation
import nltk
nltk.data.path.append('/usr/local/nltk_data')
stop_words = stopwords.words('indonesian')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
custom_stopwords = ['mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali', 
                    'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
all_stop_words = stop_words + custom_stopwords

# Preprocessing functions
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_text(text):
    text = clean_text(text)
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stop_words]
    return ' '.join(words)

# File path
MODEL_PATH = 'model.pkl'

# Train or load model
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("Model loaded from file.")
        return joblib.load(MODEL_PATH)

    print("Training model...")
    query = "SELECT description, component FROM training_data"
    df = pd.read_sql(query, engine)
    df.drop_duplicates(inplace=True)
    df['description'] = df['description'].str.lower().apply(preprocess_text)

    X = df['description']
    y = df['component']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = make_pipeline(TfidfVectorizer(stop_words=all_stop_words), MultinomialNB(alpha=1.0))
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved. Accuracy:", model.score(X_test, y_test))
    return model

# Prediction helper
def predict_component(model, description):
    description = preprocess_text(description)
    probas = model.predict_proba([description])[0]
    components = model.classes_
    predictions = {components[i]: np.round(probas[i] * 100, 2) for i in range(len(components))}
    return dict(sorted(predictions.items(), key=lambda item: item[1]))