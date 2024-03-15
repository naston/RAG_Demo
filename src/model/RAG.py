import dspy
from instruct import LanguageModel


class RAG(dspy.Module):
    def __init__(self, num_passages=10):
        super().__init__()
        
        #self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate_answer(context=context, question=question)
        return answer
    
    def retrieve(question:str):
        # Use embeddor to embed
        pass


if __name__=='__main__':
    LM = LanguageModel("google/gemma-2b-it")
    dspy.settings.configure(lm=LM)