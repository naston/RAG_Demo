from preprocess.embed import EmbedModel
from preprocess.search import VectorIndex
from model.instruct import DSPyLanguageModel
from model.retreive import DSPyRetrieverModel
from model.RAG import RAG
import dspy

import json
import numpy as np
from preprocess.parse import parse_folder

def parse_test(embed):
    with open('./data/doc_map.json','r+') as f:
        doc_map = json.load(f)
        doc_map = parse_folder('./data/00_test/','./data/02_processed/','./data/03_vectors/', doc_map, embed_model=embed)

        f.seek(0)
        json.dump(doc_map, f)
    f.close()
    return doc_map

def create_index(xb):
    index = VectorIndex()
    index.create_index_exact(xb.shape[1])
    index.add_vectors(xb)
    return index

def main():
    EM = EmbedModel('hkunlp/instructor-large')
    #index = VectorIndex()
    vector_store = np.load('./data/03_vectors/vector_store.npy')
    index = create_index(vector_store)
    doc_map = parse_test(EM)

    f = open("./access_token.txt", "r")
    access_token = f.read()
    f.close()
    
    RM = DSPyRetrieverModel(EM, index, doc_map)
    LM = DSPyLanguageModel("google/gemma-2b-it", access_token=access_token)
    dspy.settings.configure(lm=LM, rm=RM)
    rag = RAG()
    #query = 'What does the acronym S4 stand for and why is this model important?'
    query = "What equation in used to define a state space model?"

    print(query)
    print()
    print(rag(query).answer)


if __name__=='__main__':
    main()