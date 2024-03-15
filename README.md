# RAG_Demo
This repository acts as a guideline for RAG implementation. It uses open-source embedding and language models hosted on hugging-face, and implements ANN locally.

## TODO:
### Stage 1: The Backbone of RAG
- Parising Code
- Embedding Code
- Search Code
- Basic Search Test for errors

### Stage 2: Generating Chat
- Choose LLM and load it - https://huggingface.co/google/gemma-2b-it
- LM code
- Test Basic Chat 

### Stage 3: Retrieve and Generate
- Run DSPy for RAG <- You are here!
- Repurpose code for DSPy (https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve/faiss_rm.py,https://github.com/stanfordnlp/dspy/blob/649ba32fc04e864b1036edeb8ae6d330cdcc5ac7/dsp/modules/lm.py,https://github.com/stanfordnlp/dspy/blob/649ba32fc04e864b1036edeb8ae6d330cdcc5ac7/dsp/modules/hf.py)
- process all docs
- pre-train an index
- create a front end?

### Final Thoughts:
- Security (Data Access, Logins, adversarial prompting)
- Hosting (Multiple Users, non-local)
- UX (Non-technical users likely don't want a terminal)


Table of chunk index, text id, doc id