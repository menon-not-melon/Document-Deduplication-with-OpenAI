# Document Clustering and Deduplication with OpenAI Integration

This Python script automates document clustering and deduplication tasks using machine learning techniques and integrates with OpenAI for advanced deduplication capabilities.

## Overview

The script achieves the following objectives:

- **Document Preprocessing**: Utilizes NLTK for text preprocessing tasks such as tokenization, stop word removal, and lemmatization.
- **Document Clustering**: Implements TF-IDF vectorization and KMeans clustering to group similar documents together.
- **Deduplication with OpenAI**: Leverages OpenAI's API to extract the unique content from clustered documents, eliminating duplicate content.
- **Output Creation**: Produces DOCX files containing consolidated information for each detected company.

## Requirements

Ensure you have the following prerequisites:

- Python 3.x
- Required Python libraries (`nltk`, `docx2txt`, `scikit-learn`, `openai`)
- Valid API key for OpenAI

## Usage

1. **Navigate to Your Project Directory**:

    Open Command Prompt and change to the directory where you want to create your virtual environment
    ```
    cd path\to\your\project
    ```

2. **Create the Virtual Environment**:

    ```bash
    python -m venv env
    ```

3. **Activate the Virtual Environment**:

    ```bash
    .\env\Scripts\activate
    ```

4. **Install Dependencies**:

    ```bash
    pip3 install -r requirements.txt
    ```

2. **Set up file_paths.txt**
    - Include the file paths of the sample input files.
    - Each line should have only one file path 
    - For example: Team 4 - Project 4/Sample Inputs/Doc_1.docx

3. **Run the Script from the stored directory**

    ```bash
    python document_deduplication.py
    ``` 

4. **Follow the prompts**
    - Enter your OpenAI API key as requested.
    - Provide the file path to a text file containing paths of DOCX files to process.

5. **Output**
    - The script will generate DOCX files named after detected companies, each containing a consolidated document of the unique information along with a similarity score between the input documents and output document.

## Additional Notes

- **Logging:** Errors and warnings are logged to the console for debugging purposes.
- **Optimization:** Determines the optimal number of clusters using silhouette scores for KMeans clustering.
- **Similarity Check:** Calculates cosine similarity between the consolidated document and original documents for validation.