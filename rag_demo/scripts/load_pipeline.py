import qdrant_client
from llama_index.llms import Ollama
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore

def load_model(model='mixtral'):
    llm = Ollama(model=model)
    service_context = ServiceContext.from_defaults(llm=llm,embed_model="local") 

    return service_context

def load_vector_store(data_path, collection_name='papers'):
    client = qdrant_client.QdrantClient(
        path="./qdrant_data"
    )
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    return vector_store


def load_index(data_path, collection_name='papers',model='mixtral'):
    service_context = load_model(model)
    vector_store = load_vector_store(data_path, collection_name)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store,service_context=service_context)

    return index