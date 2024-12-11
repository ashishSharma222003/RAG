from llama_index.core import StorageContext,load_index_from_storage,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from CustomRetriever import CustomRetriever
# Retrievers
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
 
#jina-embeddings-v3: Multilingual Embeddings | dimension=1048
embed_model = HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v3")
Settings.embed_model=embed_model
#setting up Hugging face LLM Zephyr-7b-alpha
llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha")
Settings.llm=llm


storage_context=StorageContext.from_defaults(persist_dir="./storage")
storage_context_key=StorageContext.from_defaults(persist_dir="./storage_key")

#loading index from storage for vectorindex and for keywordindex
vector_index = load_index_from_storage( storage_context=storage_context)
keyword_index = load_index_from_storage( storage_context=storage_context_key)


vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)


#Hybrid search combining keyword and semantic search techniques
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

# define response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

step_decompose_transform = StepDecomposeQueryTransform(llm, verbose=True)
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        LLMRerank(##Reranker
            choice_batch_size=5,
            top_n=2,
        )
    ]
)
#query decomposition convert q question into 3 sub query
query_engine = MultiStepQueryEngine(
    custom_query_engine, query_transform=step_decompose_transform
)


