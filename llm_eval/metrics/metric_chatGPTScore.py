import re
import metric_helper as mH
from openai import OpenAI

def generate_prompt(gen_answer: str, ref_answer: str):

    return f"""
        Please assess the degree to which the generated answer captures the essential information provided in the reference answer. 
        Both context and reference answer are provided. Your task is to assign a rating between 0.00 (irrelevant) 
        and 1.00 (important) to indicate the similarity of information between them.
        Please read the texts carefully and consider your response thoroughly!

        Your response should be a FLOAT value between 0.00 and 1.00 (in increments of 0.05) and nothing else! VERY IMPORTANT!

        Generated Answer:
        {gen_answer}

        Reference Answer:
        {ref_answer}
"""


def delete_repeating_pattern(text):
    muster = [r'\n', r'\s+', r',+', r'\.+']
    muster_regexp = '|'.join(muster)
    cleaned = re.sub(r'(' + muster_regexp + r')\1+', r'\1', text)
    return cleaned

def call_chatGPT_model(gen_answer, ref_answer):

    client = OpenAI(api_key='sk-proj-tDUa47lrOTXdnyfL6FTcT3BlbkFJOdMnlJI3t7F4aTz9zw8I')
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{
        "role": "system",
        "content": generate_prompt(delete_repeating_pattern(gen_answer), delete_repeating_pattern(ref_answer))
        }
     ])
    
    if completion.usage is not None:
        return (completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens)
    else:
        return (completion.choices[0].message.content, None, None)

def convert_to_float(string):
    try:
        return float(string)
    except ValueError:
        return 0

def get_assert(output, context):

    model = output["model"]

    response, prompt_tokens, completion_tokens = call_chatGPT_model(output["output"], output["answer"])

    if prompt_tokens is None:
        prompt_tokens = 0
    if completion_tokens is None:
        completion_tokens = 0

    price_per_million_inputTokens = 0.5
    price_per_million_outputTokens = 1.5
    tokens_amount = 1000000

    price_prompt_tokens = round((prompt_tokens * price_per_million_inputTokens) / tokens_amount, 10)
    price_completion_tokens = round((completion_tokens * price_per_million_outputTokens) / tokens_amount, 10)

    price = round(price_prompt_tokens + price_completion_tokens, 10)

    price_prompt_tokens_str = format(price_prompt_tokens, '.10f')
    price_completion_tokens_str = format(price_completion_tokens, '.10f')
    price_str = format(price, '.10f')

    score = convert_to_float(response)

    return mH.metricHelper.get_metric_output(output, context, model, "chatGPTScore", score,{"price_prompt_tokens":price_prompt_tokens_str,
                                                                                            "price_completion_tokens":price_completion_tokens_str,
                                                                                            "price_str":price_str}, threshold=0.0)





get_assert({"model": "chatGPT", "output": "This is a test", "answer": "This is a test"}, "context")