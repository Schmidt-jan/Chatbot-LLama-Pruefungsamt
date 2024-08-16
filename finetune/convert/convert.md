1. Clone Llama.cpp: `git clone --recursive https://github.com/ggerganov/llama.cpp`
2. `make clean -C llama.cpp`
3. `make all -j -C llama.cpp`


# Convert merged LORA Adapter into Base Model to gguf via llama.cpp 

**(bf16!!!! -> outtype should match the dtype of the models weights (i.e. the dtype it was trained on))**

`python llama.cpp/convert_hf_to_gguf.py unload --outfile model/mistral-rag-instruct.gguf --outtype bf16`


`./llama.cpp/llama-quantize ./model/mistral-rag-instruct.gguf ./model/mistral-rag-instruct.Q4.gguf Q4_K_M`

`./llama.cpp/llama-quantize ./model/mistral-rag-instruct.gguf ./model/mistral-rag-instruct.Q8.gguf Q8_0`


# LoRA Adapter to gguf

- this converts the base model and the LoRA Adapter into gguf format respectively and merges them into each other after

## convert base model

`python llama.cpp/convert_hf_to_gguf.py /home/tpllmws23/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/d8cadc02ac76bd617a919d50b092e59d2d110aff --outfile lora/mistral7bv0.3.gguf --outtype bf16`

## convert LoRA Adapter

`python llama.cpp/convert_lora_to_gguf.py checkpoints/raft-v2/checkpoint-250  --outfile lora/mistral-rag-instruct-lora.gguf --outtype bf16 --base /home/tpllmws23/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/d8cadc02ac76bd617a919d50b092e59d2d110aff`


## merge base model and LoRA Adapter



`./llama.cpp/llama-export-lora -m lora/mistral7bv0.3.gguf --lora lora/mistral-rag-instruct-lora.gguf -o lora/merged-model-f16.gguf` 
