from langchain.llms.llamacpp import LlamaCpp
from typing import Type
from inspect import isclass

class ClassExtender:
    '''Class to temporarily extend a class with methods or overwrite its methods

    
    Example Usage:

        >>> class C:
                pass

        >>> def get_class_name(self):
            ... return self.__class__.__name__

        >>> with ClassExtender(C, get_class_name):
            ...    c = C()
            ...    print(c.get_class_name()) # prints 'C'


    '''
    def __init__(self, obj, method):
        method_name = method.__name__
        setattr(obj, method_name, method)
        self.obj = obj
        self.method_name = method_name

    def __enter__(self):
        return self.obj

    def __exit__(self, type, value, traceback):
        # removes extension method after context exit (after with block)
        delattr(self.obj, self.method_name)

class LlamaCppOutputExtender(ClassExtender):
    '''Convenience Class for ClassExtender to extend output of an LLM provided by Langchain

    Langchain only outputs the generated answer (a string) when invoking a chain (e.g. prompting the LLM),
    this usually being result.generation[0]['text'].
    Sometimes it is of importance to retrieve the metadata associated with a generation e.g. logprobs or token usage
    This class can be used to override the invoke (by default, or any other) function of Langchain's LlamaCpp wrapper to directly access
    all of the llm output.
    It will temporarily (this being inside the with block) overwrite (or extend the class with) a method.

    Example Usage (default overwrite of invoke):

        >>> from langchain.llms.llamacpp import LlamaCpp

        >>> llm = LlamaCpp(...) # instantiate LlamaCpp with some args, in most cases its already available somehow, otherwise instantiate it inside with block

        >>> with ClassExtender(LlamaCpp):
            ...    print(llm.invoke("Wie gehts?")) # prints full LLM output as json object

    Example Usage (custom method):

        >>> from langchain.llms.llamacpp import LlamaCpp

        >>> llm = LlamaCpp(...) # instantiate LlamaCpp with some args

        >>> def get_class_name(self):
            ... return self.__class__.__name__

        >>> with ClassExtender(LLamaCpp, get_class_name):
            ...    print(llm.get_class_name()) # prints 'LlamaCpp'

    '''
    def __init__(self, obj: Type[LlamaCpp], method = None):
        if obj is not LlamaCpp:
            raise ValueError(
                "Class to extend is expected to be of type LlamaCpp (from langchain.llms.llamacpp import LlamaCpp), received"
                f" Class or object of type {str(obj) if isclass(obj) else type(obj)}."
            )
        print('HI')
        if method is None:
            # use custom invoke method by default
            method = invoke
        super().__init__(obj, method)

def invoke(self, prompt: str):
    return self.client(prompt=prompt, **self._get_parameters())