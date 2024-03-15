from InstructorEmbedding import INSTRUCTOR

def get_embedor(model_name):
    model = INSTRUCTOR(model_name)
    def embed_text(text):
        instruction = "Represent the following text:"
        return model.encode([[instruction,text]])
    return embed_text


if __name__=='__main__':
    model_name = 'hkunlp/instructor-large'
    text = 'Here is some sample text to embed. Please embed it if you can.'

    embed_func = get_embedor(model_name)
    
    embedding = embed_func(text)
    print(embedding.shape)
    print(type(embedding))