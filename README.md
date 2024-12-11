# RAG with Hybrid Search

This project demonstrates the integration of multiple retrievers (keyword and semantic search) in **LlamaIndex** for advanced query processing, using a custom retrieval mechanism and query decomposition. It utilizes a combination of `VectorStore` and `KeywordTable` indices to achieve hybrid search, integrating semantic and keyword-based retrieval.

### Key Features:
- **Hybrid Search**: Combines vector-based semantic search and keyword-based search.
- **Query Decomposition**: Decomposes user queries into sub-queries for more granular and accurate search results.
- **Post-Processing with Reranking**: Applies LLM-based reranking on retrieved results to improve answer quality.
- **Evaluation Metrics**: Includes relevancy, correctness, and faithfulness evaluators for assessing response quality.
- **Streamlit Interface**: Displays correctness, relevancy scores, and latency for each query in a user-friendly interface.
- **OCR Processing**: Converts PDFs to readable text using OCR technology.
- **Document Embedding**: Converts PDF documents into vector embeddings and keyword-based indices for search.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Setup and Installation](#setup-and-installation)
3. [Usage](#usage)
    - [OCR Processing](#ocr-processing)
    - [Document Embedding](#document-embedding)
4. [Streamlit Interface](#streamlit-interface)
5. [Evaluation](#evaluation)


---

## Requirements

- Python 3.12.5
- LlamaIndex (install via `pip`)
- Hugging Face Transformers
- Ollama SDK
- Streamlit (for the web interface)
- Tesseract OCR (for PDF text extraction)
- `ocrmypdf` (for processing PDFs with OCR)
- Other dependencies specified in `requirements.txt`

---

## Setup and Installation

### 1. Clone the Repository:
```bash
git clone https://github.com/ashishSharma222003/RAG.git
cd RAG
```

### 2. Create and Activate Virtual Environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages:
You can install the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Install Ollama:
Ollama is used for the LLM in this project. Follow these steps to install and set it up:

- Go to the [Ollama download page](https://ollama.com/download) and download the appropriate version for your operating system.
- After installation, open **PowerShell** (or your terminal) and run the following command to pull the model:
  ```bash
  ollama pull llama3.2:3b
  ```
  This will download the `llama3.2:3b` model from Ollama.

- Once the model is downloaded, you need to run the following command to start the Ollama server:
  ```bash
  ollama serve
  ```
  This command will start the Ollama model server, allowing you to use the `llama3.2:3b` model for processing queries in your application.

### 5. Install Tesseract OCR:
- The `ocr.py` script uses **Tesseract OCR** to process PDF files. Install Tesseract by following the instructions for your OS:
  - **Windows**: [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`

- Ensure that the Tesseract executable is added to your system's `PATH`.


### 6. Download and Set Up Embeddings:
- The Hugging Face model (`BAAI/bge-small-en-v1.5`) is used for semantic search. Ensure you have access to it or modify the embedding model in the code to match your preferences.
- You can change the embedding model in the code by editing the `embed_model` initialization to use a different Hugging Face model if required.

### 8. Configure Your Indices:
Ensure your indices (`vector_index` and `keyword_index`) are properly configured and stored. Modify the `persist_dir` paths if needed to match your setup.

---

## Usage

### OCR Processing (`ocr.py`)

If you have PDF documents that need to be processed and converted into readable text, use the **OCR** (Optical Character Recognition) functionality provided in the `ocr.py` script. This script will process all PDF files from a source directory, run OCR on them, and save the output in a destination directory as readable PDF files.

#### Steps to Use `ocr.py`:

1. **Set Up Source and Destination Directories**:
   - The source directory (`./non_ocr_pdf`) should contain all the PDFs that need OCR processing.
   - The destination directory (`./data`) is where the readable PDFs will be saved.

2. **Run the Script**:
   To start OCR processing, run the following command:

   ```bash
   python ocr.py
   ```

   This will:
   - Scan all PDFs in the `./non_ocr_pdf` directory.
   - Perform OCR on each PDF using Tesseract OCR and save the output as a new PDF in the `./data` directory.
   - After processing, the original PDF file will be deleted.

   **Languages Supported**: The OCR will process documents in multiple languages, including:
   - English (`eng`)
   - Hindi (`hin`)
   - Bengali (`ben`)
   - Simplified Chinese (`chi_sim`)
   - Traditional Chinese (`chi_tra`)

3. **Note**: If the PDFs are already OCR-processed, the `ocr.py` script will forcefully reprocess them with the `force_ocr=True` flag.

### Document Embedding (`document_embeder.py`)

Once the PDFs have been processed using OCR, you can convert these readable PDFs into **vector embeddings** and **keyword-based indices** using the `document_embeder.py` script. This script uses the **BAAI/bge-small-en-v1.5** embedding model to create embeddings and store them in the respective directories.

#### Steps to Use `document_embeder.py`:

1. **Run the Script**:
   To convert your OCR-processed PDFs into embeddings and keywords, execute the following command:

   ```bash
   python document_embeder.py
   ```

   This will:
   - Process the PDFs in the `./data` directory (OCR-processed files).
   - Generate **vector embeddings** using Hugging Face embeddings.
   - Generate **keyword indices** for semantic search.
   - Save the embeddings in `./storage` (vector embeddings) and `./storage_key` (keyword embeddings).

2. **Indexing Process**:
   - The script loads the OCR-processed documents from the `./data` directory.
   - It then splits the documents into chunks using the `TokenTextSplitter` transformation, which ensures each chunk fits within the token limit of the embedding model.
   - The embeddings are generated using the **BAAI/bge-small-en-v1.5** Hugging Face model and stored in `./storage` and `./storage_key` directories.
   - These embeddings will be used later for performing search and retrieval using LlamaIndex.

### Directory Structure After Embedding:
- The `document_embeder.py` script will save two types of indices:
  - **Vector Embeddings**: Stored in the `./storage` directory.
  - **Keyword Embeddings**: Stored in the `./storage_key` directory.

These indices will be loaded during the query processing stage for hybrid search functionality.

---

## Streamlit Interface

The project includes a **Streamlit app** (`app.py`) that provides an interactive interface for users to query the engine and see performance metrics such as correctness, relevancy, and latency.

### Running the Streamlit App:
To run the Streamlit app, simply execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open a local web interface where you can:
- Enter a query.
- View the generated response.
- See the **correctness**, **relevancy**, and **latency** metrics for each query.

The Streamlit interface provides a user-friendly way to interact with the query engine and visualize the quality and performance of the responses.

---

## Evaluation

The response quality can be evaluated using three different evaluators:
1. **Relevancy Evaluator**: Assesses how relevant the response is to the original query.
2. **Correctness Evaluator**: Compares the response to a reference answer for factual accuracy.
3. **Faithfulness Evaluator**: Ensures that the response maintains faithfulness to the original context and query.

You can use these evaluators by calling the respective functions, for example:

```python
# Relevancy evaluation
relevancy_score = relevancy("Why is Blue Ocean Strategy important?", response)

# Correct

ness evaluation
correctness_score = correctness(response, "Why is Blue Ocean Strategy important?", "Reference Answer")

# Faithfulness evaluation
faithfulness_score = faithfullnes(response)
```

---

## Project Structure

```
.
├── chat.py                    # Main script to run the query engine
├── app.py                     # Streamlit app for displaying correctness, relevancy, and latency
├── ocr.py                     # OCR script to convert PDFs to readable text
├── document_embeder.py        # Script to convert PDFs into vectors and keywords
├── CustomRetriever.py          # Custom hybrid retriever combining keyword and vector retrieval
├── requirements.txt            # Project dependencies
├── storage                     # Stored vector index data
├── storage_key                 # Stored keyword index data
└── README.md                   # Project documentation
```
---
## Limitations and Considerations

- **Multilingual Embeddings**: Although the project uses multilingual embeddings, the model was primarily trained for **English**. The multilingual embeddings for **Hindi**, **Bengali**, and **Chinese** were not used, as I encountered issues running multilingual open-source models. Currently, only **English** is supported for semantic search.
  
- **Vector Database Integration**: The system does not integrate with high-performance vector databases such as **Pinecone** or **Weaviate** for large-scale data handling. While the current setup works for smaller datasets, it would be advisable to integrate with these vector databases for production-level applications and large datasets.


---

### Notes:
- **Ollama Server**: The `ollama serve` command must be running in order to process queries.
