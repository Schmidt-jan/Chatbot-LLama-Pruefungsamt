# Supervised Fine Tuning

- using huggingface trl `SFT Trainer`

- with a dataset consisting of messages or the following instruction format there is no need to preprocess it (e.g. tokenization, padding to same length, inserting special tokens e.g. INST for mistral)
- will under the hood apply the default chat template of the model

```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```


- uses by default the  [DataCollatorForLanguageModeling](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling) with mlm set to False
    - this means that no special mask is used and the labels are the same as the inputs (only padding tokens are ignored by setting them to -100) -> mlm will randomly mask tokens
    - labels are shifted by one for next token prediction (auto-regressive) task
    - prompts are fed into the model as context, the models prediction is evaluated against the expected completion
        -> cross-entropy loss function
    - next token prediction objective, aber mit der instruction (-> bei simplem prompting wird die instruction also mit predicted)
    - SFTTrainer of TRL trains the model to predict both instructions and completions.

```
For example, if your dataset is structured with different input/label (completion) pairs like this:

##input: "The dog barked at the"

##label: "cat"

Then the model will receive the entire string "the dog barked at the cat" as input, and then effectively "self supervise" it's learning by continually looking at the next token, with the preceding tokens as context.

    The -> dog

    The dog -> barked

    The dog barked -> at

    The dog barked at -> the

    The dog barked at the -> cat

For each of these examples above, the model will predict the text right of the arrow using the context left of the arrow. The loss will be based on how far off that prediction is from the actual token. 
```


**DataCollatorForCompletionOnly**

- trains on completions only -> instruction is masked away with -100 


```
DataCollatorForLanguageModeling(tokenizer=LlamaTokenizerFast(name_or_path='mistralai/Mistral-7B-v0.3', vocab_size=32768, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
	0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	3: AddedToken("[INST]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	4: AddedToken("[/INST]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	5: AddedToken("[TOOL_CALLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	6: AddedToken("[AVAILABLE_TOOLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	7: AddedToken("[/AVAILABLE_TOOLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	8: AddedToken("[TOOL_RESULTS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	9: AddedToken("[/TOOL_RESULTS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	10: 


```



