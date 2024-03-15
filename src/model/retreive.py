import numpy as np
import dspy
from dsp.utils import dotdict

class RetrieverModel(object):
    def __init__(self, embed, index) -> None:
        self.embed = embed
        self.index = index

    def __call__(self, query, k=10):
        query_vect = self.embed(query)
        D, I = self.index.search_vectors(query_vect, k)
        
        context = None
        return context
    
class _RetrieverModel(dspy.Retrieve):
    def __init__(self, embed, index, doc_map):
        self.embed = embed
        self.index = index
        self.doc_map = doc_map

    def get_document(self, doc_id):
        pass

    def forward(self, query, k=3):
        """Search the faiss index for k or self.k top passages for query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        query_vect = self.embed(query)
        distances, indices = self.index.search_vectors(query_vect, k*3)

        passage_scores = {}
        for i in range(len(indices[0])):
            doc_id = indices[0][i]
            doc_dist = distances[0][i]
            passage_scores[doc_id] = doc_dist
            
        sorted_passages = sorted(passage_scores.items(), key=lambda x: x[1])[:k]
        return [ dotdict({"long_text": self.get_document(passage_index), "index": passage_index}) for passage_index, _ in sorted_passages ]
    

if __name__=='__main__':
    RM = RetrieverModel(None,None)