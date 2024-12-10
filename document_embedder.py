from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
num_workers=1

# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.embed_model=embed_model

documents = SimpleDirectoryReader(input_dir="./data").load_data(show_progress=True,num_workers=num_workers)
transformations = [
    SentenceSplitter(chunk_size=512, chunk_overlap=128),
    TitleExtractor(nodes=5),
]


index=VectorStoreIndex.from_documents(documents=documents,show_progress=True,transformations=transformations)
index.storage_context.persist()



