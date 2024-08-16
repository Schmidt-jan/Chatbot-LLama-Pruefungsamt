# Prototype / RAG Demo


- currently it is only possible to prompt openai and local llms in the gguf format via the prototype

## Installation

1. You will need `llama-cpp-python` and our local rag lib installed (see [here](../README.md#installation) for details)
2. `pip install -r requirements.txt`
3. `cd frontend && npm install`


## Start Prototype

1. Start backend (running on Port `8000`): `python3 backend/server.py`
2. Start frontend (running on Port `3000`): `cd frontend && npm start`