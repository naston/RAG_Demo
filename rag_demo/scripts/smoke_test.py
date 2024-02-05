# Just runs .complete to make sure the LLM is listening
from llama_index.llms import Ollama

llm = Ollama(model="tinyllama")

print('generate response...')

response = llm.complete("Who is Lionel Messi?")
print(response)