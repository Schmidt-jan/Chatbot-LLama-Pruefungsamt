from helper.capture_output import CaptureOutput
from helper.class_extender import LlamaCppOutputExtender
from helper.llamacpp_timings import LlamaCppTimings
from helper.perplexity import compute_perplexity

def call_llm(model: str, rag_chain, prompt, options, context):
    answer = context['vars']['answer']

    output = None
    from langchain.llms.llamacpp import LlamaCpp
    with CaptureOutput() as capture, LlamaCppOutputExtender(LlamaCpp):
        output = rag_chain(prompt, answer)

    print(output)


    timings = LlamaCppTimings.from_string(capture.stderr.getvalue())

    print('\nTIMINGS: \n')
    print(timings.to_json())

    print('\nPerplexity: \n')
    print(compute_perplexity(output['choices'][0]['logprobs']['token_logprobs']))

    result = {
        'output': {
            'model': model,
            'output': output['choices'][0]['text'],
            'answer': answer,
            'perplexity': compute_perplexity(output['choices'][0]['logprobs']['token_logprobs']),
            'timings': timings.to_dict(),
            'token_usage': output['usage'],
            'finish_reason': output['choices'][0]['finish_reason']
        }
    }

    return result