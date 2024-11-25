import os
import docx2txt
import nltk
import logging
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
from openai import OpenAI
from docx import Document

# Download necessary NLTK resources
nltk.download('punkt')  #Used for tokenization
nltk.download('stopwords')  #Used to download stopword corpus to filtering them out
nltk.download('wordnet')    #Used for tasks like lemmatization (reducing words to their base or root form)

# Clear the terminal screen based on OS type
os.system('cls' if os.name == 'nt' else 'clear')

# Initialize OpenAI client with API key
key = str(input("Enter your Open AI API key: "))
client = OpenAI(api_key = key)

# Configure logging
logging.basicConfig(
    level = logging.WARNING,  # Set logging level to WARNING to only display errors and warnings
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    return " ".join(tokens)  # Return preprocessed text as string (for TF-IDF)
