from preprocess.embed import EmbedModel
from preprocess.search import VectorIndex
from model.instruct import LanguageModel
from model.retreive import RetreiverModel
from model.RAG import RAG


def main():
    EM = EmbedModel()
    index = VectorIndex()
    
    RM = RetreiverModel(EM, index)
    LM = LanguageModel()

    rag = RAG(LM, RM)
    pass


if __name__=='__main__':
    main()