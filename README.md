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
- Test Basic Chat <- You are here

### Stage 3: Retrieve and Generate
- Run DSPy for RAG
- process all docs
- pre-train an index
- create a front end?

### Final Thoughts:
- Security (Data Access, Logins, adversarial prompting)
- Hosting (Multiple Users, non-local)
- UX (Non-technical users likely don't want a terminal)