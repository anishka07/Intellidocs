# IntelliDocs

## Overview

**IntelliDocs** is a Retrieval-Augmented Generation (RAG) based project designed to assist users in querying and extracting information from their PDF documents. By leveraging advanced natural language processing techniques, IntelliDocs enables users to efficiently retrieve relevant content from large volumes of text within PDFs.

## Project Objectives

1. **PDF Extraction**: Implement methods to extract text from PDF files, ensuring the preservation of formatting and structure.
2. **Text Processing**: Clean and tokenize extracted text to prepare it for chunking and embedding.
3. **Chunking**: Divide the processed text into manageable chunks to facilitate efficient querying.
4. **Embedding**: Use Sentence Transformers to generate embeddings for the text chunks, enabling semantic similarity searches.
5. **Querying**: Develop a retrieval system that allows users to input queries and receive relevant chunks of text based on semantic similarity.
6. **Structuring**: Structure the generated response with the help of a LLM.

## Technologies Used

- **Programming Language**: `Python`
- **Libraries**:
  - `fitz`: For PDF text extraction.
  - `sentence-transformers`: For embedding text chunks.
  - `Streamlit`: For creating the user interface.
  - `Chromadb`: For vector database.

## Project Structure

```plaintext
├── README.md
├── algorithms
│   ├── __init__.py
│   └── cosine_similarity.py
├── examples
│   ├── __init__.py
│   └── intellidocs_rag_main.py
├── id_chroma_db
│   └── cache
          # contains chroma db cache
├── images
          # ui snippets
├── llms
│   ├── __init__.py
│   ├── gemini_response.py
│   └── llama_response.py
├── model
│   ├── __init__.py
│   ├── intellidocs_rag_final
│   └── tfidf
│       ├── TF-IDFSummarizer.py
│       └── __init__.py
├── pdfs
          # contains PDFs for input
├── requirements.txt
├── ui.py
└── utils
    ├── cleanup_chroma.py
    ├── constants.py
    └── dir_utils.py
```

# Step-by-Step Guide to Clone and Run IntelliDocs

## Prerequisites

Ensure you have the following installed on your system:
- Python (version 3.7 or higher)
- pip (Python package installer)
- Git

## Step 1: Clone the Repository

Open your terminal or command prompt and run the following command:

```bash
git clone https://github.com/anishka07/intellidocs.git
```

## Step 2: Create a virtual environment using conda or virtual env and activate it

Run the following command:

```bash
## Example:
conda create -n your_env_name python=3.11 pip -y
conda activate your_env_name 
```

## Step 3: Install Requirements

Run the following command:

```bash
pip install -r requirements.txt
```

## Step 4: Run IntelliDocs from terminal or streamlit

To run IntelliDocs from terminal:

```bash
cd examples
python intellidocs_rag_main.py (make sure to check the file)
```

To run IntelliDocs from it's streamlit UI:

```bash
streamlit run ui.py
```

## Streamlit Interface
User Interface:
![User Interface](images/abc.png)

Indexing multiple PDFs as input:
![](images/def.png)

Query Response (Both Structured and Relevant Chunks):
![](images/ghi.png)
## Usage

1. **Input PDF**: Upload your PDF/PDFs using the Streamlit interface.
2. **Querying**: Select the PDF you want to query using the unique generated PDF key and query the PDF.
3. **Results**: The system will return the most relevant text chunks extracted from the PDF selected.

## Future Work

- **Expand Support**: Extend support to other document formats (e.g., DOCX, TXT).
- **Web Application**: Create a full stack web application with apis.


