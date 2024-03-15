from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel(object):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    
    def generate(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**tokens)
        return self.tokenizer.decode(outputs[0])


def chat(LM:LanguageModel):
    print('What can I help you with?\n')
    response = input()
    while response!='quit' and response!='exit':
        print(LM.generate(response)+'\n')
        response = input()


if __name__=='__main__':
    LM = LanguageModel("google/gemma-2b-it")

    input_text = "Write me a poem about Machine Learning."
    response = LM.generate(input_text)
    print(response)