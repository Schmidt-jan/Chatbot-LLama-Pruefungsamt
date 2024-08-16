import sys
from io import StringIO

class CaptureOutput:
    '''Class to redirect stdout and stderr into StringIO() objects

    Example Usage:

        >>> from llama_cpp import Llama
        >>> llm = Llama(
            model_path="llms/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_batch=2048,
            max_tokens=-1,
            n_ctx=2048,
            f16_kv=True,
            verbose=True
        )

        >>> with CaptureOutput() as capture:
        ...    print("something")
        ...    prompt = "Tell me about the Roman Empire?"
        ...    reply = ""
        ...    for token in llm(prompt, stream=True, echo=False):
        ...        item = token["choices"][0]["text"]
        ...        reply += item
        ...        print(item, sep=' ', end='', flush=True)

    Access captured outputs:
        >>> print("Captured stdout:", capture.stdout.getvalue())
        >>> print("Captured stderr:", capture.stderr.getvalue())
    '''
    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr



'''
from llama_cpp import Llama

llm = Llama(
            model_path="llms/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_batch=2048,
            max_tokens=-1,
            n_ctx=2048,
            f16_kv=True,
            verbose=True
        )


with CaptureOutput() as capture:
    print("something")
    prompt = "Tell me about the Roman Empire?"
    reply = ""
    for token in llm(prompt, stream=True, echo=False):
        item = token["choices"][0]["text"]
        reply += item
        #print(token["choices"][0]["text"])
        #print(item, end=" ")
        print(item, sep=' ', end='', flush=True)

# Access captured outputs
print("Captured stdout:", capture.stdout.getvalue())
print("Captured stderr:", capture.stderr.getvalue())

'''


