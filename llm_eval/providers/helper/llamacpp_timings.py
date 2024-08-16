import re

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadTime:
    time: float

    def __post_init__(self):
       object.__setattr__(self, 'time', float(self.time))

@dataclass(frozen=True)
class SampleTime:
    time: float
    runs: int
    ms_per_token: float
    tokens_per_second: float

    def __post_init__(self):
        object.__setattr__(self, 'time', float(self.time))
        object.__setattr__(self, 'runs', int(self.runs))
        object.__setattr__(self, 'ms_per_token', float(self.ms_per_token))
        object.__setattr__(self, 'tokens_per_second', float(self.tokens_per_second))

@dataclass(frozen=True)
class PromptEvalTime:
    time: float
    tokens: int
    ms_per_token: float
    tokens_per_second: float

    def __post_init__(self):
        object.__setattr__(self, 'time', float(self.time))
        object.__setattr__(self, 'tokens', int(self.tokens))
        object.__setattr__(self, 'ms_per_token', float(self.ms_per_token))
        object.__setattr__(self, 'tokens_per_second', float(self.tokens_per_second))

@dataclass(frozen=True)
class EvalTime:
    time: float
    runs: int
    ms_per_token: float
    tokens_per_second: float

    def __post_init__(self):
        object.__setattr__(self, 'time', float(self.time))
        object.__setattr__(self, 'runs', int(self.runs))
        object.__setattr__(self, 'ms_per_token', float(self.ms_per_token))
        object.__setattr__(self, 'tokens_per_second', float(self.tokens_per_second))

@dataclass(frozen=True)
class TotalTime:
    time: float
    tokens: int

    def __post_init__(self):
       object.__setattr__(self, 'time', float(self.time))
       object.__setattr__(self, 'tokens', int(self.time))

@dataclass(frozen=True)
class LlamaCppTimings():
    '''Class for storing timings printed out by LlamaCpp when prompting a model

    Load Time: time it takes to load the model (file)

    Sample Time: 
        - time for choosing the next likely token
        - "sampling" time means running the RNG, sorting and filtering candidates

    Prompt Eval Time:
        - how long it took to process the prompt (before generating an answer/text)^
        - e.g. prompt eval time =    1166.27 ms /     8 tokens (  145.78 ms per token,
            -> 8 tokens where evaluated in 1166.27 ms
            -> 6.86 tokens per second (1000 ms / 145.78)

    Eval Time: 
        - time it takes to generate output (answer)

    Timings are parsed from a string like the following:
    
        llama_print_timings:        load time =    1166.42 ms
        llama_print_timings:      sample time =       4.46 ms /    16 runs   (    0.28 ms per token,  3590.66 tokens per second)
        llama_print_timings: prompt eval time =    1166.27 ms /     8 tokens (  145.78 ms per token,     6.86 tokens per second)
        llama_print_timings:        eval time =     431.82 ms /    15 runs   (   28.79 ms per token,    34.74 tokens per second)
        llama_print_timings:       total time =    1634.10 ms /    23 tokens
    '''
    load_time: LoadTime
    sample_time: SampleTime
    prompt_eval_time: PromptEvalTime
    eval_time: EvalTime
    total_time: TotalTime

    @classmethod
    def from_string(cls, timingOutput: str) -> "LlamaCppTimings":
        timing_tuple = _parse_timing_string(timingOutput)
        return cls(*timing_tuple)
    
    def to_json(self):
        import json, dataclasses
        return json.dumps(dataclasses.asdict(self), indent=4)
    
    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)


def _parse_timing_string(string):
    load_time_pattern = r"load time =\s+([\d.]+) ms"
    sample_time_pattern = r"sample time =\s+([\d.]+) ms \/\s+(\d+) runs\s+\(\s+([\d.]+) ms per token,\s+([\d.]+) tokens per second\)"
    prompt_eval_time_pattern = r"prompt eval time =\s+([\d.]+) ms \/\s+(\d+) tokens\s+\(\s+([\d.]+) ms per token,\s+([\d.]+) tokens per second\)"
    eval_time_pattern = r"eval time =\s+([\d.]+) ms \/\s+(\d+) runs\s+\(\s+([\d.]+) ms per token,\s+([\d.]+) tokens per second\)"
    total_time_pattern = r"total time =\s+([\d.]+) ms \/\s+(\d+) tokens"

    load_time_match = re.search(load_time_pattern, string)
    sample_time_match = re.search(sample_time_pattern, string)
    prompt_eval_time_match = re.search(prompt_eval_time_pattern, string)
    eval_time_match = re.search(eval_time_pattern, string)
    total_time_match = re.search(total_time_pattern, string)

    load_time = LoadTime(load_time_match.group(1)) if load_time_match else None
    sample_time = SampleTime(*sample_time_match.groups()) if sample_time_match else None
    prompt_eval_time = PromptEvalTime(*prompt_eval_time_match.groups()) if prompt_eval_time_match else None
    eval_time = EvalTime(*eval_time_match.groups()) if eval_time_match else None
    total_time = TotalTime(*total_time_match.groups()) if total_time_match else None

    return load_time, sample_time, prompt_eval_time, eval_time, total_time
    

    




'''
# Example usage:
timing_string = """
llama_print_timings:        load time =    1166.42 ms
llama_print_timings:      sample time =       4.46 ms /    16 runs   (    0.28 ms per token,  3590.66 tokens per second)
llama_print_timings: prompt eval time =    1166.27 ms /     8 tokens (  145.78 ms per token,     6.86 tokens per second)
llama_print_timings:        eval time =     431.82 ms /    15 runs   (   28.79 ms per token,    34.74 tokens per second)
llama_print_timings:       total time =    1634.10 ms /    23 tokens
"""


timings = LlamaCppTimings.from_string(timing_string)

'''



'''
load_time, sample_time, prompt_eval_time, eval_time, total_time = parse_timing_string(timing_string)

print("Load Time:", load_time.time)
print("Sample Time:", sample_time.time, sample_time.runs, sample_time.ms_per_token, sample_time.tokens_per_second)
print("Prompt Eval Time:", prompt_eval_time.time, prompt_eval_time.tokens, prompt_eval_time.ms_per_token, prompt_eval_time.tokens_per_second)
print("Eval Time:", eval_time.time, eval_time.runs, eval_time.ms_per_token, eval_time.tokens_per_second)
print("Total Time:", total_time.time, total_time.tokens)
'''

'''
# Example usage:
timing_string = """
llama_print_timings:        load time =    1166.42 ms
llama_print_timings:      sample time =       4.46 ms /    16 runs   (    0.28 ms per token,  3590.66 tokens per second)
llama_print_timings: prompt eval time =    1166.27 ms /     8 tokens (  145.78 ms per token,     6.86 tokens per second)
llama_print_timings:        eval time =     431.82 ms /    15 runs   (   28.79 ms per token,    34.74 tokens per second)
llama_print_timings:       total time =    1634.10 ms /    23 tokens
"""


timings = LlamaCppTimings.from_string(timing_string)


print(timings.to_json())

print(timings.to_dict())
'''
