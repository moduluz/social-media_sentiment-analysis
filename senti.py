!pip install tqdm nltk scikit-learn pandas numpy

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from time import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None


class SentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags
        text = re.sub(r'#\w+', '', text)

        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in self.stop_words and len(token) > 2]

        return ' '.join(tokens)

    def load_and_prepare_data(self, filepath, sample_size=None):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(filepath, names=column_names, encoding='ISO-8859-1')

        if sample_size:
            df = df.sample(n=sample_size, random_state=42)

        df['target'] = df['target'].replace(4, 1)

        print("Preprocessing texts...")
        tqdm.pandas()
        df['processed_text'] = df['text'].progress_apply(self.preprocess_text)

        return df

    def prepare_features(self, texts, max_features=10000):
        """Convert text to TF-IDF features"""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=max_features,
                                            ngram_range=(1, 2))
            return self.vectorizer.fit_transform(texts)
        return self.vectorizer.transform(texts)

    def train_model(self, X_train, y_train):
        """Train the sentiment analysis model"""
        print("Training model...")
        self.model = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and print metrics"""
        predictions = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    def save_model(self, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Save the trained model and vectorizer"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")

    def load_model(self, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load a trained model and vectorizer"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print("Model and vectorizer loaded successfully")

    def predict(self, texts):
        """Predict sentiment for new texts"""
        if isinstance(texts, str):
            texts = [texts]

        processed_texts = [self.preprocess_text(text) for text in texts]
        features = self.prepare_features(processed_texts)
        predictions = self.model.predict(features)
        labels = ['Negative' if pred == 0 else 'Positive' for pred in predictions]

        return labels if len(labels) > 1 else labels[0]

from google.colab import files

# Upload your kaggle.json file
files.upload()  # Upload your kaggle.json here

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Install and use kaggle
!pip install kaggle
!kaggle datasets download -d kazanova/sentiment140

# Extract the dataset
!unzip -q sentiment140.zip

# Download all necessary NLTK resources
import nltk

# Download all required resources in a single cell
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
print("All NLTK resources downloaded successfully")

# Verify the downloads
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
print("\nVerifying downloads:")
print("Stopwords available:", len(stopwords.words('english')), "words")
print("Tokenization test:", word_tokenize("This is a test sentence."))

# Create analyzer instance
analyzer = SentimentAnalyzer()

# Load and prepare data
df = analyzer.load_and_prepare_data(
    'training.1600000.processed.noemoticon.csv',
    sample_size=100000
)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'],
    df['target'],
    test_size=0.2,
    random_state=42,
    stratify=df['target']
)

# Prepare features
print("Preparing features...")
X_train_features = analyzer.prepare_features(X_train)
X_test_features = analyzer.prepare_features(X_test)

# Train the model
analyzer.train_model(X_train_features, y_train)

# Evaluate model performance
analyzer.evaluate_model(X_test_features, y_test)

# Save model and vectorizer for future use
analyzer.save_model()

# Test with example tweets
example_tweets = [
    "I absolutely love this new product! It's amazing!",
    "This is the worst experience ever. Terrible service.",
    "The weather is nice today."
]

print("\nTesting with example tweets:")
predictions = analyzer.predict(example_tweets)
for tweet, prediction in zip(example_tweets, predictions):
    print(f"\nTweet: {tweet}")
    print(f"Sentiment: {prediction}")

# Add probability scores to predictions
def predict_with_confidence(texts):
    """Predict sentiment with confidence scores"""
    if isinstance(texts, str):
        texts = [texts]

    processed_texts = [analyzer.preprocess_text(text) for text in texts]
    features = analyzer.prepare_features(processed_texts)

    # Get probabilities instead of just predictions
    probabilities = analyzer.model.predict_proba(features)
    predictions = analyzer.model.predict(features)

    results = []
    for text, prob, pred in zip(texts, probabilities, predictions):
        confidence = prob[1] if pred == 1 else prob[0]
        sentiment = 'Positive' if pred == 1 else 'Negative'
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence * 100  # Convert to percentage
        })
    return results

# Test with various challenging examples
test_tweets = [
    # Sarcasm
    "Oh great, another fantastic day of meetings 🙄",

    # Mixed sentiment
    "The graphics are amazing but the storyline is terrible",

    # Subtle negativity
    "Well, I've seen worse I suppose",

    # Subtle positivity
    "It's not perfect, but it grows on you",

    # Ambiguous
    "This is different from what I expected",

    # Context-dependent
    "This movie made me cry",

    # Neutral statement
    "Just finished watching the whole series",

    # Conditional positive
    "If you like action movies, you'll probably enjoy this"
]

# Get predictions with confidence scores
results = predict_with_confidence(test_tweets)

# Print results with detailed analysis
print("\nDetailed Sentiment Analysis:")
print("-" * 80)
for result in results:
    print(f"\nTweet: {result['text']}")
    print(f"Predicted Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.1f}%")

    # Add analysis for potentially problematic cases
    confidence = result['confidence']
    if confidence < 60:
        print("Note: Low confidence prediction - might be ambiguous or need context")
    elif confidence < 75:
        print("Note: Moderate confidence - might contain mixed signals")

    if '?' in result['text'] or '!' in result['text']:
        print("Note: Contains punctuation that might affect sentiment")
    if '🙄' in result['text'] or '😊' in result['text'] or '😢' in result['text']:
        print("Note: Contains emoji that might indicate sarcasm or additional context")

