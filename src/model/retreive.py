import numpy as np

class RetreiverModel():
    def __init__(self, embed, index) -> None:
        self.embed = embed
        self.index = index

    def forward(self, query):
        query_vect = self.embed(query)
        D, I = self.index.search_vectors(query_vect)
        
        context = None
        return context