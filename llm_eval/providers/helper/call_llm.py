from enum import Enum

from helper.capture_output import CaptureOutput
from helper.class_extender import LlamaCppOutputExtender
from helper.llamacpp_timings import LlamaCppTimings
from helper.perplexity import compute_perplexity
import json

import psutil
import time
from contextlib import contextmanager

class USE_DB(Enum):
    TRUE = True
    FALSE = False
    
@contextmanager
def monitor_power(model):
    # Start monitoring
    process = psutil.Process()
    start_time = time.time()
    start_energy = process.cpu_times().user + process.cpu_times().system

    yield

    # End monitoring
    end_time = time.time()
    end_energy = process.cpu_times().user + process.cpu_times().system
    elapsed_time = end_time - start_time
    energy_used = end_energy - start_energy

    # Assuming an average CPU power usage of 15 watts (this can vary widely)
    average_cpu_power_watts = 15
    watt_hours = (energy_used * average_cpu_power_watts) / 3600  # converting from seconds to hours

    print(f"[{model}] Energy used: {watt_hours:.5f} watt-hours")

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


def call_llm_huggingface(model: str, rag_chain, prompt, options, context):
    answer = context['vars']['answer']

    output = None
    documents = []

    documents, output = rag_chain(prompt, answer, USE_DB.TRUE)


    try:
        perplexity = compute_perplexity(output['choices'][0]['logprobs']['token_logprobs'])
    except:
        perplexity = None

    result = {
        'output': {
            'model': model,
            'output': output,
            'answer': answer,
            'keywords': context['vars']['keywords']['list'],
            'perplexity': perplexity,
            'timings': None,
            'token_usage': None,
            'finish_reason': None,
            'context': json.dumps([ob.__dict__ for ob in documents])
        }
    }

    return result