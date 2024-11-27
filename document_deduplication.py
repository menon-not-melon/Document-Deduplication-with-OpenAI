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

# Function to read DOCX files from a list of file paths and preprocess them
def read_and_preprocess_docx_files(file_paths):
    documents = []
    for file_path in file_paths:
        text = docx2txt.process(file_path.strip())  # Extract text from DOCX
        processed_text = preprocess_text(text)  # Preprocess extracted text
        documents.append(processed_text)  # Add preprocessed text to list
    return documents

# Function to read DOCX files from a list of file paths without preprocessing
def read_docx_files(file_paths):
    documents = []
    for file_path in file_paths:
        text = docx2txt.process(file_path.strip())  # Extract text from DOCX
        documents.append(text)  # Add original text to list
    return documents

# Function to summarize topics using OpenAI's API
def summarize_topic(text):
    summaries = []  #Initializes an empty list summaries to store the results of the API call

    #Used to create a chat-based completion response from the API.
    response = client.chat.completions.create(      
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant."   #Provides background information or instructions to the model
            },
            {
                "role": "assistant",   #Contains the prompt for the model. Prompt can be edited to suit different input cases
                "content": f"Find the company based on these extracts from these documents: [ {text} ]. From these extracts, remove the duplicate sentences and consolidate into a single output. Start the answer with the company name and Deduplicated Extracts directly. No need to bold or italicise any text, output must be in format 'Company Name=' and 'Extract='. All extracts after deduplication must be included",
            }
        ],
        model="gpt-4o-mini"
    )
    #Appends the response from the API to the summaries list
    summaries.append(response.choices[0].message)       
    return summaries
