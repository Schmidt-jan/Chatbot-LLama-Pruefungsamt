# Chatbot-LLama-Pruefungsamt

In diesem Repository sind die Skripte und Ergebnisse des Teamprojekts 'Ein Chatbot für das Prüfungsamt'.

Die [Abschlusspräsentation](./Abschlussbericht/praesentation.pptx) und die eine kurze [Zusammenfassung]() sind in dem Ordner [Abschlussbericht/](./Abschlussbericht/) zu finden.

## Datenbasis
Als Datenbasis wurden die Dokumente [Zulassungssatzung für die Masterstudiengänge (ZuSMa)](./main_data_filtered/119_ZuSMa_Senat_18012022.pdf) und [SPO Nr. 5 - Studiengang Informatik (MSI)](./main_data_filtered/SPO_MSI_SPONr5_Senat_10122019.pdf) verwendet.

## Libraries
In dem Ordner [libs](./libs/) sind einige Libraries enthalten welche wir erstellt haben.   
Die Bibliothek [custom_rag_loader](./libs/custom_rag_loader/) haben wir dazu verwendet um die Datenbank sowie die LLMs zu laden.

## Evaluierung - Retrieval Augmented Generation (RAG)
Für die Evaluierung des Retrieval Augmented Generation Ansatzes sind die folgenden Ordner wichtig:
- [Retrieval](./Retrieval/) - Alles was mit dem Erstellen der Datenbasis und deren Tests zu tun hat
- [Evaluierung des RAGs]()
- [prototype](./prototype/) - Ein Prototyp eines einfachen Chatbots

## Finetuning
Alles zu dem Finetuning ist in dem Ordner [finetune](./finetune/) zu finden.


![](.github/assets/chatbot-logo.jpeg)



## Installation


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
