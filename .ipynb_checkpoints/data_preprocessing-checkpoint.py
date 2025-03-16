import pandas as pd
import numpy as np
import re
import nltk
import tensorflow_hub as hub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """Cleans and tokenizes text, removing stopwords."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize words
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

def load_and_preprocess_data(file_path):
    """Loads the dataset, removes NaN values, and applies text preprocessing."""
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df

def convert_to_embeddings(df):
    """Converts preprocessed text into embeddings using TensorFlow Hub."""
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    X = np.array([embed([text]).numpy()[0] for text in df['cleaned_text']])
    y = df['label'].values  # Assuming 'label' is the column for fake/real review
    return X, y
