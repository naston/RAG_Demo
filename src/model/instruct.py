from transformers import AutoTokenizer, AutoModelForCausalLM
from dsp.modules.lm import LM
import torch


class LanguageModel(object):
    def __init__(self, model_name, access_token=None) -> None:
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token).to(self.device)

    
    def __call__(self, text:str):
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**tokens,max_new_tokens=1028)

        input_length = tokens.input_ids.shape[1]
        outputs = outputs[:, input_length:-1]

        return self.tokenizer.decode(outputs[0])
    

class _LanguageModel(LM):
    def __init__(self, model_name, access_token=None) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def _generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs,max_new_tokens=256)
        
        input_length = inputs.input_ids.shape[1]
        outputs = outputs[:, input_length:]

        completions = [{"text": c} for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        response = {
            "prompt": prompt,
            "choices": completions,
        }
        return response
    
    def __call__(self, prompt:str):
        response = self.request(prompt)
        return [c["text"] for c in response["choices"]]


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

    #input_text = "What team does Lionel Messi play for?"
    input_text = "What is your name?"
    print(input_text)
    print()
    response = LM(input_text)
    print(response)