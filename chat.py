# Import necessary libraries from llama_index and transformers
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    SimpleKeywordTableIndex,
    VectorStoreIndex,
    get_response_synthesizer,
    set_global_tokenizer,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, MultiStepQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
from transformers import AutoTokenizer
from CustomRetriever import CustomRetriever

# Initialize embedding model (Jina embeddings with BGE model)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True)
Settings.embed_model = embed_model

# Initialize LLM (using Ollama's Llama3.2 model)
llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
Settings.llm = llm

# Evaluation functions
def relevancy(query: str, response):
    evaluator = RelevancyEvaluator(llm=llm)
    return evaluator.evaluate_response(query=query, response=response)

def correctness(response: str, query: str, reference: str):
    evaluator = CorrectnessEvaluator(llm=llm)
    return evaluator.evaluate(query=query, response=response, reference=reference)

def faithfullnes(response):
    evaluator = FaithfulnessEvaluator(llm=llm)
    return evaluator.evaluate_response(response=response)

# Load storage context and indices
storage_context = StorageContext.from_defaults(persist_dir="./storage")
storage_context_key = StorageContext.from_defaults(persist_dir="./storage_key")

# Load the indices (Vector and Keyword)
vector_index = load_index_from_storage(storage_context=storage_context)
keyword_index = load_index_from_storage(storage_context=storage_context_key)

print("Index loaded successfully.")

# Setup retrievers
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)

print("Retrievers set up successfully.")

# Hybrid search combining both keyword and semantic search
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)
print("Custom retriever initialized.")

# Define response synthesizer (for tree summarization)
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# Setup query decomposition and custom query engine
step_decompose_transform = StepDecomposeQueryTransform(llm, verbose=True)
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        LLMRerank(choice_batch_size=5, top_n=2)  # Reranker configuration
    ]
)

print("Query engine setup completed.")

# Setup multi-step query engine (query decomposition into sub-queries)
query_engine = MultiStepQueryEngine(
    custom_query_engine, 
    query_transform=step_decompose_transform,
    index_summary="You are the best giver."  # Optional context for summarization
)

print("Multi-step query engine initialized.")

# Main execution block
if __name__ == "__main__":
    response = query_engine.query("Why Is Blue Ocean Strategy of Rising Importance?")
    print("Query Response: ", response)
