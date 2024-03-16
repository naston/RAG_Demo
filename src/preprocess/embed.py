from typing import Any
from InstructorEmbedding import INSTRUCTOR
import numpy as np

def get_embedor(model_name):
    model = INSTRUCTOR(model_name)
    def embed_text(text):
        instruction = "Represent the following text:"
        return model.encode([[instruction,text]])
    return embed_text


class DummyEmbedModel(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, text:str):
        return np.array([0,0,0,0])

class EmbedModel(object):
    def __init__(self,model_name) -> None:
        super().__init__()
        self.model = INSTRUCTOR(model_name)

    def __call__(self, text:str):
        instruction = "Represent the following text:"
        return self.model.encode([[instruction,text]])


if __name__=='__main__':
    model_name = 'hkunlp/instructor-large'
    text = 'Here is some sample text to embed. Please embed it if you can.'

    embed_func = get_embedor(model_name)
    
    embedding = embed_func(text)
    print(embedding.shape)
    print(type(embedding))