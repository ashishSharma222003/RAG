from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    get_response_synthesizer,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, MultiStepQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from CustomRetriever import CustomRetriever

import os

def initialize_query_engine(
    user_id: str,
    base_dir: str = "./user_data",
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    llm_model_name: str = "llama3.2:3b",
    similarity_top_k: int = 10
):
    """
    Initializes a query engine for a specific user by automatically detecting index paths.

    Args:
        user_id (str): Unique identifier for the user.
        base_dir (str): Base directory where user-specific indices are stored. Default is './user_data'.
        embedding_model_name (str): Name of the HuggingFace embedding model.
        llm_model_name (str): Name of the LLM model.
        similarity_top_k (int): Number of top results for similarity search.

    Returns:
        MultiStepQueryEngine: A query engine ready to process user queries.
    """
    # Dynamically locate user-specific index paths
    user_dir = os.path.join(base_dir, user_id)
    vector_index_path = os.path.join(user_dir, "vector_index")
    keyword_index_path = os.path.join(user_dir, "keyword_index")

    if not (os.path.exists(vector_index_path) and os.path.exists(keyword_index_path)):
        raise FileNotFoundError(f"Indices for user '{user_id}' not found.")

    print(f"Using vector index path: {vector_index_path}")
    print(f"Using keyword index path: {keyword_index_path}")

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, trust_remote_code=True)
    Settings.embed_model = embed_model

    # Initialize LLM
    llm = Ollama(model=llm_model_name, request_timeout=120.0)
    Settings.llm = llm

    # Load stored indices
    vector_storage_context = StorageContext.from_defaults(persist_dir=vector_index_path)
    keyword_storage_context = StorageContext.from_defaults(persist_dir=keyword_index_path)

    vector_index = load_index_from_storage(storage_context=vector_storage_context)
    keyword_index = load_index_from_storage(storage_context=keyword_storage_context)

    print("Indices loaded successfully.")

    # Set up retrievers
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=similarity_top_k)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)

    print("Retrievers set up successfully.")

    # Combine retrievers using a custom retriever
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)
    print("Custom retriever initialized.")

    # Define response synthesizer
    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

    # Setup query decomposition and custom query engine
    step_decompose_transform = StepDecomposeQueryTransform(llm, verbose=True)
    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            LLMRerank(choice_batch_size=10, top_n=5)  # Reranker configuration
        ]
    )

    print("Query engine setup completed.")

    # Setup multi-step query engine
    # query_engine = MultiStepQueryEngine(
    #     custom_query_engine, 
    #     query_transform=step_decompose_transform,
    #     index_summary="You are the best giver."  # Optional context for summarization
    # )

    # print("Multi-step query engine initialized.")
    return custom_query_engine

if __name__=="__main__":
    query_engine =initialize_query_engine('admin')
    response=query_engine.query("describe what is chain of thoughts")
