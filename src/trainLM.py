from preprocess.embed import EmbedModel
from preprocess.search import VectorIndex
from model.instruct import DSPyLanguageModel
from model.retreive import DSPyRetrieverModel
from model.RAG import DSPyRAG
import dspy
from dspy.teleprompt import BootstrapFewShot

import json
import numpy as np
from preprocess.parse import parse_folder


def create_index(xb):
    index = VectorIndex()
    index.create_index_exact(xb.shape[1])
    index.add_vectors(xb)
    return index


def validate_context_and_answer(example, pred, trace=None):
    # check the gold label and the predicted answer are the same
    answer_match = example.answer.lower() == pred.answer.lower()

    # check the predicted answer comes from one of the retrieved contexts
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    return answer_match and context_match


def tuneLM():
    query = "Why are SSMs still slower than Transformers during inference?"

    EM = EmbedModel('hkunlp/instructor-large')
    #index = VectorIndex()
    vector_store = np.load('./data/03_vectors/vector_store.npy')
    index = create_index(vector_store)
    with open('./data/doc_map.json','r+') as f:
        doc_map = json.load(f)
    f.close()

    f = open("./access_token.txt", "r")
    access_token = f.read()
    f.close()
    
    RM = DSPyRetrieverModel(EM, index, doc_map)
    LM = DSPyLanguageModel("google/gemma-2b-it", access_token=access_token)
    dspy.settings.configure(lm=LM, rm=RM)
    rag = DSPyRAG()

    print(query)
    print()
    print(rag(query).answer)

    print()
    print('Tuning RAG...')
    with open('./data/05_tuning/dspy_tuning.json','r+') as f:
        ft_data = f.read()
    f.close()
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
    compiled_rag = teleprompter.compile(rag, trainset=ft_data)

    print()
    print(compiled_rag(query).answer)

    compiled_rag.save('./data/06_model/compiled_rag.json')


def loadLM():
    EM = EmbedModel('hkunlp/instructor-large')
    #index = VectorIndex()
    vector_store = np.load('./data/03_vectors/vector_store.npy')
    index = create_index(vector_store)
    with open('./data/doc_map.json','r+') as f:
        doc_map = json.load(f)
    f.close()

    f = open("./access_token.txt", "r")
    access_token = f.read()
    f.close()
    
    RM = DSPyRetrieverModel(EM, index, doc_map)
    LM = DSPyLanguageModel("google/gemma-2b-it", access_token=access_token)
    dspy.settings.configure(lm=LM, rm=RM)

    rag = DSPyRAG()
    rag.load('compiled_cot_gsm8k.json')

    query = "Why are SSMs still slower than Transformers during inference?"

    print(query)
    print()
    print(rag(query).answer)


def testRAG():
    EM = EmbedModel('hkunlp/instructor-large')
    #index = VectorIndex()
    vector_store = np.load('./data/03_vectors/vector_store.npy')
    index = create_index(vector_store)
    with open('./data/doc_map.json','r+') as f:
        doc_map = json.load(f)
    f.close()

    f = open("./access_token.txt", "r")
    access_token = f.read()
    f.close()
    
    RM = DSPyRetrieverModel(EM, index, doc_map)
    LM = DSPyLanguageModel("google/gemma-2b-it", access_token=access_token)
    dspy.settings.configure(lm=LM, rm=RM)
    rag = DSPyRAG()
    #query = 'What does the acronym S4 stand for and why is this model important?'
    query = "What equation in used to define a state space model?"

    print(query)
    print()
    print(rag(query).answer)


if __name__=='__main__':
    #testRAG()
    tuneLM()
    pass