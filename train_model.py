import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from joblib import dump
from sqlalchemy import create_engine

# Setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = stopwords.words('indonesian')
custom_stopwords = ['mohon', 'tolong', 'bantu', 'masalah', 'baiknya', 'berkali', 'kali', 
                    'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
all_stop_words = stop_words + custom_stopwords
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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

# DB connection
DATABASE_URI = 'mysql+pymysql://root:GcchhrdqnsKyauycgVpnYKpXMzYSELhn@ballast.proxy.rlwy.net:58414/railway?charset=utf8mb4'
engine = create_engine(DATABASE_URI)

# Fetch data
query = "SELECT description, component FROM training_data"
df = pd.read_sql(query, engine)
df.drop_duplicates(inplace=True)
df['description'] = df['description'].str.lower().apply(preprocess_text)

X = df['description']
y = df['component']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = make_pipeline(TfidfVectorizer(stop_words=all_stop_words), MultinomialNB(alpha=1.0))
model.fit(X_train, y_train)

# Save model
dump(model, 'model.joblib')
