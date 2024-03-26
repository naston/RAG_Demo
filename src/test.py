from preprocess.embed import EmbedModel
from preprocess.search import VectorIndex
from model.instruct import LanguageModel
from model.retreive import RetrieverModel
from preprocess.parse import parse_folder
import transformers
import json
import numpy as np

transformers.logging.set_verbosity_error()


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


def generic_test():
    query = 'What does the acronym S4 stand for and why is this model important?'

    # Load LM
    f = open("./access_token.txt", "r")
    access_token = f.read()
    f.close()
    LM = LanguageModel("google/gemma-2b-it", access_token=access_token)

    embed = EmbedModel('hkunlp/instructor-large')

    # Init data
    doc_map = parse_test(embed)
    vector_store = np.load('./data/03_vectors/vector_store.npy')

    # create retriever
    index = create_index(vector_store)
    RM = RetrieverModel(embed, index, doc_map)
   
    # retreive context
    context = RM(query,k=3)

    print(context)

    #test_query  = 'What is your name?'
    print()
    print(LM(query))
    print()

    # Generate Response
    prompt = "[Query]\n"+query +'\n[Context]\n'
    for c in context:
        prompt += c['long_text'] +'\n'

    response = LM(prompt)
    print(response)


if __name__=='__main__':
    generic_test()