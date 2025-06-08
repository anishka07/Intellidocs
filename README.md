# IntelliDocs

## Overview

**IntelliDocs** is a Retrieval-Augmented Generation (RAG) based project designed to assist users in querying and extracting information from their PDF documents. By leveraging advanced natural language processing techniques, IntelliDocs enables users to efficiently retrieve relevant content from large volumes of text within PDFs.

## Project Objectives

1. **PDF Extraction**: Implement methods to extract text from PDF files, ensuring the preservation of formatting and structure.
2. **Chunking**: Divide the text into manageable chunks to facilitate efficient querying.
3. **Embedding**: Use Sentence Transformers to generate embeddings for the text chunks, enabling semantic similarity searches.
4. **Querying**: Develop a retrieval system that allows users to input queries and receive relevant chunks of text based on semantic similarity.
5. **Structuring**: Structure the generated response with the help of a LLM.

## Technologies Used

- **Programming Language**: `Python`
- **Libraries**:
  - `fitz`: For PDF text extraction.
  - `sentence-transformers`: For embedding text chunks.
  - `Streamlit`: For creating the user interface.
  - `Chromadb`: For vector database.

# Step-by-Step Guide to Clone and Run IntelliDocs

## Prerequisites

Ensure you have the following installed on your system:
- Python (version 3.12)
- uv (Python package installer)
- Git

## Step 1: Clone the Repository

Open your terminal or command prompt and run the following command:

```bash
git clone https://github.com/anishka07/intellidocs.git
```

## Step 2: Create a runnable environment automatically with uv

Run the following command:

```bash
uv sync 
```

## Step 3: Run IntelliDocs using gRPC or streamlit

Run IntelliDocs gRPC server and client:

```bash
uv run python server.py 
```

```bash
uv run python client.py process *your pdf's name*

uv run python client.py query *your pdf key* *your query*
```

To run IntelliDocs from it's streamlit UI:

```bash
uv run streamlit run ui.py
```

## Streamlit Interface
User Interface:
![User Interface](resources/user_interface.png)

Indexing multiple PDFs as input:
![Indexing multiple PDFs as input](resources/index.png)

Query Response (Both Structured and Relevant Chunks):
![Query Response (Both structured and relevant chunks)](resources/query_res.png)
## Usage

1. **Input PDF**: Upload your PDF/PDFs using the Streamlit interface.
2. **Querying**: Select the PDF you want to query using the unique generated PDF key and query the PDF.
3. **Results**: The system will return the most relevant text chunks extracted from the PDF selected.

## TODOs

1. modify the gRPC code 
2. Use Enums, Pydantic and make it more dynamic
3. Web application with FastAPI