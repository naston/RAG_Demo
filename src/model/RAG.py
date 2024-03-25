import dspy


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        
        self.num_passages = num_passages
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate_answer(context=context, question=question)
        return answer
    
    def _retrieve(self, question:str):
        # Use embeddor to embed
        return self.rm(question, k=self.num_passages)


if __name__=='__main__':
    pass