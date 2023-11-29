# initialize simple vector indices
from typing import List
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from zenml import step


@step(enable_cache=True)
def index_generator2(documents: List[Document]) -> VectorStoreIndex:
    embed_model = OpenAIEmbedding()

    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    return VectorStoreIndex.from_documents(
        documents=documents, service_context=service_context
    )