from llama_index.core import StorageContext,load_index_from_storage,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM
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
# from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.llms.ollama import Ollama


#jina-embeddings-v3: Multilingual Embeddings | dimension=1048
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",trust_remote_code=True)
Settings.embed_model=embed_model
#setting up Hugging face LLM Zephyr-7b-alpha
# llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha",tokenizer_name="HuggingFaceH4/zephyr-7b-beta")
llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
Settings.llm=llm
# set_global_tokenizer(
#     AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha").encode
# )


storage_context=StorageContext.from_defaults(persist_dir="./storage")
storage_context_key=StorageContext.from_defaults(persist_dir="./storage_key")

#loading index from storage for vectorindex and for keywordindex
vector_index = load_index_from_storage( storage_context=storage_context)
keyword_index = load_index_from_storage( storage_context=storage_context_key)
print("load index")

vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)

print("***retriever***")
#Hybrid search combining keyword and semantic search techniques
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)
print("***custom retriever***")
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
print("*********"*5)
#query decomposition convert q question into 3 sub query
query_engine = MultiStepQueryEngine(
    custom_query_engine, query_transform=step_decompose_transform,index_summary="you are the best giver"
)
print('query_engine')
print("**********"*5)
if __name__=="__main__":
    response=query_engine.query("Why Is Blue Ocean Strategy of Rising Importance?")
    print(response)

