from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter,TokenTextSplitter
from llama_index.core.extractors import TitleExtractor,SummaryExtractor 
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer


num_workers=1#can't increase due pc resource constraints

llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
Settings.llm=llm
set_global_tokenizer(
    AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha").encode
)
# loads jinaai/jina-embeddings-v3
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",trust_remote_code=True)

Settings.embed_model=embed_model
print("*"*80)
documents = SimpleDirectoryReader(input_dir="./data",recursive=True).load_data(show_progress=True,num_workers=num_workers)
pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=256, chunk_overlap=128),
        # SentenceSplitter(chunk_size=256,chunk_overlap=256),
        # TitleExtractor(nodes=3),                                      #Can't use this extractor since they consume all my pc resource and starts to 
        # SummaryExtractor(), 
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
)
nodes=pipeline.run(documents=documents,num_workers=num_workers)


vector_index = VectorStoreIndex(nodes)
keyword_index = SimpleKeywordTableIndex(nodes)
# index=VectorStoreIndex.from_documents(documents=documents,show_progress=True,transformations=transformations)
vector_index.storage_context.persist(persist_dir="./storage")
keyword_index.storage_context.persist(persist_dir="./storage_key")



