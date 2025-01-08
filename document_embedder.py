import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama

def create_and_save_user_indices(
    user_id: str,
    input_dir: str, 
    base_dir: str = "./user_data",
    chunk_size: int = 512, 
    chunk_overlap: int = 256, 
    num_workers: int = 1,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    llm_model_name: str = "llama3.2:3b"
):
    """
    Create and save user-specific VectorStoreIndex and KeywordTableIndex.

    Args:
        user_id (str): Unique identifier for the user.
        input_dir (str): Directory containing documents for the user.
        base_dir (str): Base directory to store user-specific indices. Default is './user_data'.
        chunk_size (int): Size of text chunks for splitting. Default is 512.
        chunk_overlap (int): Overlap size between chunks. Default is 256.
        num_workers (int): Number of workers for parallel processing. Default is 1.
        embedding_model_name (str): Name of the HuggingFace embedding model. Default is 'BAAI/bge-small-en-v1.5'.
        llm_model_name (str): Name of the LLM model. Default is 'llama3.2:3b'.

    Returns:
        dict: Paths to the stored vector and keyword indices.
    """
    # Create user-specific directories
    user_dir = os.path.join(base_dir, user_id)
    vector_store_dir = os.path.join(user_dir, "vector_index")
    keyword_store_dir = os.path.join(user_dir, "keyword_index")
    os.makedirs(vector_store_dir, exist_ok=True)
    os.makedirs(keyword_store_dir, exist_ok=True)

    # Initialize the LLM and embedding model
    llm = Ollama(model=llm_model_name, request_timeout=120.0)
    Settings.llm = llm

    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, trust_remote_code=True)
    Settings.embed_model = embed_model

    # Load documents from the specified directory
    documents = SimpleDirectoryReader(input_dir=input_dir, recursive=True).load_data(
        show_progress=True, num_workers=num_workers
    )

    # Set up the ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)]
    )

    # Run the pipeline to process documents
    nodes = pipeline.run(documents=documents, num_workers=num_workers)

    # Create VectorStoreIndex and KeywordTableIndex
    vector_index = VectorStoreIndex(nodes)
    keyword_index = SimpleKeywordTableIndex(nodes)

    # Persist indices to user-specific directories
    vector_index.storage_context.persist(persist_dir=vector_store_dir)
    keyword_index.storage_context.persist(persist_dir=keyword_store_dir)

    print(f"Vector index saved at: {vector_store_dir}")
    print(f"Keyword index saved at: {keyword_store_dir}")

    # return {
    #     "vector_index_path": vector_store_dir,
    #     "keyword_index_path": keyword_store_dir,
    # }
