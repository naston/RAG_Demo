from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel(object):
    def __init__(self, model_name) -> None:
        super().__init__()

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")



if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

    input_text = "Write me a poem about Machine Learning."
    tokens = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**tokens)
    output_text = tokenizer.decode(outputs[0])
    print(output_text)