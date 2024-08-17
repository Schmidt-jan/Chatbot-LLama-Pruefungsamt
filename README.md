# Chatbot-LLama-Pruefungsamt


![](.github/assets/chatbot-logo.jpeg)


This repository contains the code and results of the team project ‘A Chatbot for the Examination Office’ at the HTWG Konstanz.


- [Retrieval](./Retrieval/)

The [final presentation](./final-report/presentation.pptx) and a short [summary](./final-report/summary.pdf) can be found in the [final-report](./final-report/) directory.



  * ### [Structure](#structure)
  * ### [Installation](#installation)
  * ### Additional README's
    * ### [🔎 Retrieval](./retrieval/README.md)
    * ### [📊 Evaluation](./llm_eval/README.md)
    * ### [✨ Finetuning](./finetune/README.md)
    * ### [🏃 Prototype](./prototype/README.md)


## Structure

### Retrieval Augmented Generation (RAG) and Evaluation
The following folders are important for running and evaluating the Retrieval Augmented Generation approach:
- [Retrieval](./retrieval/) - creation of the database from the documents and its tests
- [Custom RAG Loader](./libs/custom_rag_loader) - our small local library build on top of [Langchain](https://github.com/langchain-ai/langchain)
- [Evaluation of the RAG](./llm_eval/)
- [Prototype](./prototype/) - A prototype/demo of a simple chatbot

#### Data basis
The documents [Zulassungssatzung für die Masterstudiengänge (ZuSMa)](./main_data_filtered/119_ZuSMa_Senat_18012022.pdf) and [SPO Nr. 5 - Studiengang Informatik (MSI)](./main_data_filtered/SPO_MSI_SPONr5_Senat_10122019.pdf) were used as the data basis and ingested in our vector database.

### Finetuning


We finetuned [Mistral7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) using several approaches and datasets. More information and Notebooks can be found in the [finetune](./finetune/) subdirectory.

## Installation

As most of the libraries used in the context of Large Language Models are mostly undergoing rapid development with a lot of API changes we recommend installing the dependencies with the version specified in the respective `requirements.txt`.

### Prerequisites

- python wrapper for [LLama.cpp](https://github.com/ggerganov/llama.cpp): `pip install CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade llama-cpp-python==0.2.84` (flags for non CUDA setups can be found in [their](https://github.com/abetlen/llama-cpp-python) installation instructions)
- our local rag lib: `pip install ./libs/custom_rag_loader/`
- (optionally jupyter and ipython notebooks)
- [Promptfoo](https://github.com/promptfoo/promptfoo) for llm_eval (we used version `0.54.1` if latest does not work): `npx promptfoo@latest` or `npm install -g promptfoo@latest`


### Install Python Dependencies

- `pip install -r requirements.txt`

There are separate `requirements.txt` for [Finetuning](./finetune/) and the [Prototype](./prototype/).
They can be optionally installed aswell with (or installed independently if only one of those topics is of interest)

`pip install -r ./finetune/requirements.txt` 

and

`pip install -r ./prototype/requirements.txt`


We recommend creating separate for finetuning and rag (e.g. llm_eval or running the prototype) to avoid version/dependency conflicts:

```
cd ~/environments

python3 -m venv <env-name>

source ~/environments/<env-name>/bin/activate
```


### Adding Models

- we hardcoded (sorry) the model paths and model_file_names in our [custom_rag_loader](libs/custom_rag_loader/custom_rag_loader.py)
- they need to be adjusted [here](libs/custom_rag_loader/custom_rag_loader.py?plain=1#L14-15) and [here](libs/custom_rag_loader/custom_rag_loader.py?plain=1#L103-128)