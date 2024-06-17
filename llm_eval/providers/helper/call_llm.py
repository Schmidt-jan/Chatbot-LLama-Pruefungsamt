from enum import Enum

from sympy import per
from helper.capture_output import CaptureOutput
from helper.class_extender import LlamaCppOutputExtender
from helper.llamacpp_timings import LlamaCppTimings
from helper.perplexity import compute_perplexity
import json

class USE_DB(Enum):
    TRUE = True
    FALSE = False
    

def call_llm(model: str, rag_chain, prompt, options, context):
    answer = context['vars']['answer']

    output = None
    documents = []
    from langchain.llms.llamacpp import LlamaCpp
    with CaptureOutput() as capture, LlamaCppOutputExtender(LlamaCpp):
        documents, output = rag_chain(prompt, answer, USE_DB.TRUE)

    # print(output)
    #print(f"DOCUMENTS: {documents}")

    timings = LlamaCppTimings.from_string(capture.stderr.getvalue())

    # print('\nTIMINGS: \n')
    # print(timings.to_json())
    # print('\nPerplexity: \n')
    # print(compute_perplexity(output['choices'][0]['logprobs']['token_logprobs']))

    try:
        perplexity = compute_perplexity(output['choices'][0]['logprobs']['token_logprobs'])
    except:
        perplexity = None

    result = {
        'output': {
            'model': model,
            'output': output['choices'][0]['text'],
            'answer': answer,
            'keywords': context['vars']['keywords']['list'],
            'perplexity': perplexity,
            'timings': timings.to_dict(),
            'token_usage': output['usage'],
            'finish_reason': output['choices'][0]['finish_reason'],
            'context': json.dumps([ob.__dict__ for ob in documents])
        }
    }

    return result