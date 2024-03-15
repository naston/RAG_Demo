import numpy as np
import dspy
from dsp.utils import dotdict

class RetrieverModel(object):
    def __init__(self, embed, index) -> None:
        self.embed = embed
        self.index = index

    def __call__(self, query):
        query_vect = self.embed(query)
        D, I = self.index.search_vectors(query_vect)
        
        context = None
        return context
    
class _RetrieverModel(dspy.Retrieve):
    def __init__(self, document_chunks, embed, index):
        self.embed = embed
        self.index = index

        self.document_chunks = document_chunks

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
            doc_id = indices[i]
            doc_dist = distances[i]
            if doc_id in passage_scores:
                    passage_scores[doc_id].append(doc_dist)
            else:
                passage_scores[doc_id] = [doc_dist]
        sorted_passages = sorted(passage_scores.items(), key=lambda x: (1 - len(x[1]), sum(x[1])))[:k]
        return [ dotdict({"long_text": self.document_chunks[passage_index], "index": passage_index}) for passage_index, _ in sorted_passages ] #This is where I will need to spend some more time tbh