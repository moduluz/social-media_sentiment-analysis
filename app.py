# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load NLTK resources (run only once, or ensure they are downloaded)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data - ENSURE THESE ARE DOWNLOADED
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt_tab', quiet=True) # ADDED THIS LINE AGAIN TO BE SURE

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def prepare_features(self, texts):
        """Convert text to TF-IDF features"""
        if self.vectorizer is None:
            # Load vectorizer if it's not already loaded (should happen during app initialization)
            self.load_vectorizer()
        return self.vectorizer.transform(texts)

    def predict_with_confidence(self, texts):
        """Predict sentiment with confidence scores"""
        if isinstance(texts, str):
            texts = [texts]

        processed_texts = [self.preprocess_text(text) for text in texts]
        features = self.prepare_features(processed_texts)

        probabilities = self.model.predict_proba(features)
        predictions = self.model.predict(features)

        results = []
        for text, prob, pred in zip(texts, probabilities, predictions):
            confidence = prob[1] if pred == 1 else prob[0]
            sentiment = 'Positive' if pred == 1 else 'Negative'
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence * 100
            })
        return results

    def load_model_and_vectorizer(self, model_path='models/sentiment_model.pkl', vectorizer_path='models/vectorizer.pkl'):
        """Load a trained model and vectorizer"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully from file.")
        except FileNotFoundError:
            error_message = f"Model file not found at: {model_path}"
            print(error_message)
            raise FileNotFoundError(error_message) # Raise exception

        self.load_vectorizer(vectorizer_path) # Call load_vectorizer, which now also raises exceptions


    def load_vectorizer(self, vectorizer_path='models/vectorizer.pkl'):
        """Load only the vectorizer"""
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Vectorizer loaded successfully from file.")
        except FileNotFoundError:
            error_message = f"Vectorizer file not found at: {vectorizer_path}"
            print(error_message)
            raise FileNotFoundError(error_message) # Raise exception to halt execution
        except Exception as e:
            error_message = f"Error loading vectorizer from {vectorizer_path}: {e}"
            print(error_message)
            raise Exception(error_message) # Raise other exceptions


    def train_and_save_model(self, filepath='training.1600000.processed.noemoticon.csv', model_path='models/sentiment_model.pkl', vectorizer_path='models/vectorizer.pkl'):
        """Train the model and save it along with the vectorizer."""
        print("Training model...")
        column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(filepath, names=column_names, encoding='ISO-8859-1')
        df['target'] = df['target'].replace(4, 1)
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        X_train, _, y_train, _ = train_test_split(
            df['processed_text'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
        )

        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train_features = self.vectorizer.fit_transform(X_train)

        self.model = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
        self.model.fit(X_train_features, y_train)

        # Save model and vectorizer
        os.makedirs(os.path.dirname(model_path), exist_ok=True) # Ensure directory exists
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model and vectorizer saved to {model_path} and {vectorizer_path}")


# app.py

# Initialize SentimentAnalyzer and load model and vectorizer when the app starts
analyzer = SentimentAnalyzer()
analyzer.load_model_and_vectorizer() # Load pre-trained model at app startup
print(f"Analyzer object initialized at startup. ID: {id(analyzer)}") # DEBUG PRINT 4


@app.route('/', methods=['GET', 'POST'])
def index():
    print(f"Index route handler called. Analyzer object ID: {id(analyzer)}") # DEBUG PRINT 5
    if request.method == 'POST':
        text = request.form['text']
        if text:
            results = analyzer.predict_with_confidence(text)
            return render_template('index.html', results=results)
    return render_template('index.html', results=None)



@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    results = analyzer.predict_with_confidence(text)
    return jsonify(results=results)


if __name__ == '__main__':
    # Train and save model if model files are not present (first time run)
    if not os.path.exists('models/sentiment_model.pkl') or not os.path.exists('models/vectorizer.pkl'):
        print("Model or vectorizer not found, training new model...")
        analyzer.train_and_save_model() # Train and save model for the first time

    app.run(debug=True) # Set debug=False in production