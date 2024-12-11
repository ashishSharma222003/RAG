# Import necessary modules from LlamaIndex and HuggingFace
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.extractors import TitleExtractor, SummaryExtractor
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer

# Set the number of workers (constrained by available PC resources)
num_workers = 1  # Can't increase due to PC resource constraints

# Initialize the LLM (Ollama model)
llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
Settings.llm = llm


# Load the embedding model (HuggingFace BGE model)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True)
Settings.embed_model = embed_model

# Print separator for clarity in the logs
print("*" * 80)

# Load documents from the specified directory
documents = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data(show_progress=True, num_workers=num_workers)

# Set up the IngestionPipeline with required transformations
pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=512, chunk_overlap=256),  # Split text into chunks of size 512 with 256 overlap
        # SentenceSplitter(chunk_size=256, chunk_overlap=256),  # Sentence-based splitting (disabled)
        # TitleExtractor(nodes=3),  # Disabled due to high resource usage
        # SummaryExtractor(),  # Disabled due to high resource usage
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),  # Embedding model for document transformation
    ]
)

# Run the ingestion pipeline to process documents
nodes = pipeline.run(documents=documents, num_workers=num_workers)

# Create the VectorStoreIndex and KeywordTableIndex
vector_index = VectorStoreIndex(nodes)
keyword_index = SimpleKeywordTableIndex(nodes)

# Persist the indices to storage
vector_index.storage_context.persist(persist_dir="./storage")
keyword_index.storage_context.persist(persist_dir="./storage_key")

# End of the script
