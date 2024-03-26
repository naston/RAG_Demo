import dspy


class DSPyRAG(dspy.Module):
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


class RAG():
    def __init__(self, LM, RM, k=3):
        self.LM = LM
        self.RM = RM
        self.k = k
        

    def __call__(self, query):
        context = self.RM(query,k=self.k)

        prompt = "[Query]\n"+query +'\n[Context]\n'
        for c in context:
            prompt += c['long_text'] +'\n'

        response = self.LM(prompt)

        return response

if __name__=='__main__':
    pass