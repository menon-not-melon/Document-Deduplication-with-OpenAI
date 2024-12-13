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

# Function to create a DOCX file with specified content
def create_docx(file_name, content):       #Takes input of the file name string and the content string

    #Creates a new Document object representing a new, empty word document from the python-docx library
    doc = Document()       
    
    # Add content to the document
    doc.add_paragraph(content)
    
    # Save the document
    doc.save("Output/" + file_name + ".docx")

# Function to remove non-alphanumeric characters from a string
def remove_non_alphanumeric(input_string):
    return ''.join(char for char in input_string if char.isalnum())

# Main execution starts here
while True:
    # Read the text file that contains file paths of input documents
    input_path = input("Enter file path of text file containing file paths of input files, \nFor E.g: C:/Users/Downloads/file_paths.txt \n\nEnter file path: ")
    docx_paths_file = input_path.replace('\\', '/')

    try:
        # Check if the file exists and is accessible
        if os.path.exists(docx_paths_file):
            with open(docx_paths_file, 'r') as f:   #Opens the file in read mode 
                file_paths = f.readlines()  #Reads all lines from the file into a list
                file_paths = [path.strip() for path in file_paths]  # Remove newline characters

                # Check each file path in file_paths
                all_paths_valid = True  #Used to track whether all file paths are valid
                for path in file_paths:  
                    if not os.path.exists(path):       #Checks if each path exists using os.path.exists
                        all_paths_valid = False
                        logging.error(f"Error: File path '{path}' does not exist or is not accessible.")
                        break

                if all_paths_valid:
                    break  # Exit the loop if all paths are valid
                else:
                    logging.warning("Please enter a new docx_paths_file with valid file paths.")
        else:
            logging.error(f"Error: File '{docx_paths_file}' does not exist or is not accessible. Please try again.")
            
    except IOError as e:
        # Raised when an I/O operation such as opening or reading a file fails
        logging.error(f"Error: Unable to open file '{docx_paths_file}'. IOError: {e}")  
    except Exception as e:
        #Catches any exceptions that are derived from the base Exception class
        logging.error(f"Error: An unexpected error occurred: {e}")

# Consolidate all documents into list of preprocessed text
documents = read_and_preprocess_docx_files(file_paths)
original_docs = read_docx_files(file_paths)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000,  # Limit the number of features to the top 1000 based on their TF-IDF scores.
                             stop_words='english', # Exclude common English stop words which do not contribute much to the analysis.
                             max_df=0.8, # Ignore terms that appear in more than 80% of the documents
                             min_df=2)  # Include terms that appear in at least 2 documents.

# Fit and transform documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Define range of k values to test for KMeans clustering
k_values = range(2, 50)  # Test for k from 2 to 49

# Initialize list to store silhouette scores for each k
silhouette_scores = []

# Iterate over each value of k
for k in k_values:  #Specifies the number of clusters
    # Fit KMeans clustering model
    kmeans = KMeans(n_clusters=k,   #Specifies the number of clusters (k) to form.
     random_state=42)  #Sets the seed for the random number generator to ensures that the results are reproducible. 
    kmeans.fit(tfidf_matrix)    #Each row in this matrix represents a document and each column represents a term, with the values indicating the importance of each term in the document.
    # Calculate silhouette score if more than one cluster is present
    if len(np.unique(kmeans.labels_)) > 1:
        # Compute the average silhouette score for the clustering
        silhouette_avg = silhouette_score(tfidf_matrix, kmeans.labels_)
        # Append the silhouette score to a list of scores
        silhouette_scores.append(silhouette_avg)
    else:
        silhouette_scores.append(-1)  # Placeholder for single cluster scenario

# Determine optimal number of clusters based on highest silhouette score
optimal_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because k_values start from 2

# Clear the terminal screen again
os.system('cls' if os.name == 'nt' else 'clear')

# Print optimal number of clusters based on Silhouette method
print(f'\nOptimal number of clusters based on Silhouette method: {optimal_clusters}')
print()

# Initialize KMeans with optimal number of clusters and fit the data
kmeans = KMeans(n_clusters=optimal_clusters,     #Specifies the number of clusters
 random_state=42)     #Sets the seed for the random number generator used in the initialization of cluster centroids.
kmeans.fit(tfidf_matrix)

# Retrieve cluster labels
cluster_labels = kmeans.labels_

# Map each document path to its cluster label

# Initialize an empty dictionary to map file paths to their respective cluster labels
document_cluster_map = {}
# Loop through each file path and its index in the list of file paths
for idx, file_path in enumerate(file_paths):
    # Strip any leading/trailing whitespace from the file path and map it to the cluster label
    document_cluster_map[file_path.strip()] = cluster_labels[idx]

# Initialize defaultdict to store documents for each cluster
cluster_documents = defaultdict(list)

# Organize documents into clusters based on labels
for idx, file_path in enumerate(file_paths):
    cluster_label = document_cluster_map[file_path.strip()]  # Get cluster label
    document_content = original_docs[idx]  # Get original content of the document

    # Append document content to corresponding cluster
    cluster_documents[cluster_label].append(document_content)

# Convert defaultdict to regular dictionary for easier manipulation
cluster_documents = dict(cluster_documents)

# Iterate through each cluster and process documents
for cluster_label, documents in cluster_documents.items():
    # Concatenate all documents into one large string
    join_docs = " ".join(documents)
    
    # Tokenize into sentences
    sentences = sent_tokenize(join_docs)

    # Initialize set to store unique sentences
    unique_sentences = []
    seen_sentences = set()

    # Remove duplicates and store unique sentences
    for sentence in sentences:
        # Check if the sentence has not been encountered before
        if sentence not in seen_sentences:
            # Add the unique sentence to the list of unique sentences
            unique_sentences.append(sentence)
            # Record the sentence as seen by adding it to the set
            seen_sentences.add(sentence)

    # Summarize topic using OpenAI API
    summary = summarize_topic(unique_sentences)

    # Extract company name and summary section from API response
    # Check if the word "Extract" is present in the content of the first summary item
    if "Extract" in summary[0].content:
        # Split the content at the first occurrence of "Extract"
        company_name_section, summary_section = summary[0].content.split("Extract", 1)
        # Extract the part of the content before "Extract" and split it to find "Company Name"
        company_name = company_name_section.split("Company Name")[1].strip()
        # Remove non-alphanumeric characters from the company name
        company_name = remove_non_alphanumeric(company_name)
    else:
        # Print an error message if the delimiter "Extract" is not found in the content
        print("Error: Expected delimiter '\n\nExtract=' not found in content.")

    # Generate filename and create a DOCX file with summarized content
    file_name = company_name + "_Consolidated_Output"
    create_docx(file_name, summary_section[1:])  # Exclude the initial delimiter

    # Print saved message
    print(f"Saved {file_name}.docx")
    print()
    
    # Combine all unique sentences into a single string
    cluster_text = " ".join(unique_sentences)
    # Preprocess the combined text 
    cluster_processed_text = preprocess_text(cluster_text)
    # Transform the summary section into a TF-IDF vector
    extract_tfidf = vectorizer.transform([summary_section])
    # Transform the preprocessed cluster text into a TF-IDF vector using the vectorizer
    cluster_tfidf = vectorizer.transform([cluster_processed_text])

    # Checking the similarity score between consolidated doc and input docs involved
    similarity_score_perc = cosine_similarity(extract_tfidf, cluster_tfidf)[0][0] * 100

    print(f"Similarity Score for {file_name}.docx : {similarity_score_perc:.2f}%")
    print("-" * 50)
    print()
