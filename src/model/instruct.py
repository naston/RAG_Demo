from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel(object):
    def __init__(self, model_name, access_token=None) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

    
    def __call__(self, text:str):
        tokens = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**tokens,max_new_tokens=256)
        return self.tokenizer.decode(outputs[0])


def chat(chat_model):
    print('What can I help you with?\n')
    response = input()
    while response!='quit' and response!='exit':
        print(chat_model(response)+'\n')
        response = input()


if __name__=='__main__':
    f = open("./access_token.txt", "r")
    access_token = f.read()
    f.close()

    LM = LanguageModel("google/gemma-2b-it",access_token=access_token)

    input_text = "What team does Lionel Messi play for?"
    response = LM(input_text)
    print(response)