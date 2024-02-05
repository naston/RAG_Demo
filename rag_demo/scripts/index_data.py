from pathlib import Path
from .process_pdf import extract_pdf, pdf_direct

import qdrant_client
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
) #StringIterableReader,
from llama_index.llms import Ollama
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

def index_paper_text():
    #texts = extract_pdf()

    #documents = StringIterableReader().load_data(
    #    texts=texts
    #)

    documents = pdf_direct()
    print('received documents',len(documents))
    # initialize the vector store
    client = qdrant_client.QdrantClient(
        path="./qdrant_data"
    )
    vector_store = QdrantVectorStore(client=client, collection_name="papers")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print('init client')

    # initialize the LLM
    llm = Ollama(model="tinyllama")
    service_context = ServiceContext.from_defaults(llm=llm,embed_model="local")

    print('init LM')

    # create the index; this will embed the documents and store them in the vector store
    index = VectorStoreIndex.from_documents(
        documents,
        llm=llm,
        storage_context=storage_context
    )

    print('Index made')

    query_engine = index.as_query_engine()

    query="""What does the acronym S4 stand for?"""

    response = query_engine.query(query)
    print(response)
