import json
from math import exp
from typing import Any, List, Optional, Tuple
from collections.abc import Mapping
from pyparsing import Opt 

class Provider:
    id: str

    def __init__(self, id: str):
        self.id = id

class Prompt:
    raw: str
    display: str

    def __init__(self, raw: str, display: str):
        self.raw = raw
        self.display = display

class Vars:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Response:
    output: str

    def __init__(self, output: str):
        self.output = output


class TokenUsed:
    total: int
    prompt: int
    completion: int

    def __init__(self, total: int, prompt: int, completion: int):
        self.total = total
        self.prompt = prompt
        self.completion = completion


class Assertion:
    def __init__(self, type: str, value: Optional[str] = None, threshold: Optional[float] = None, provider: Optional[str] = None, metric: Optional[str] = None):
        self.type = type
        self.value = value
        self.threshold = threshold
        self.provider = provider
        self.metric = metric

    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> 'Assertion':
        json_obj = json.loads(json_str)
        return cls(**json_obj)
    
class ComponentResult:
    pass_: bool
    score: float
    reason: Optional[str]
    tokens_used: Optional[TokenUsed]
    componentResults: Optional[List['ComponentResult']]
    component_results: Optional[List['ComponentResult']]
    assertion: Optional[Assertion]
    latency_ms: Optional[int]
    namedScores: Optional[Any]

    def __init__(self, pass_: bool, score: float, reason: Optional[str] = None, tokens_used: Optional[TokenUsed] = None, componentResults: Optional[List['ComponentResult']] = None, component_results: Optional[List['ComponentResult']] = None, assertion: Optional[Assertion] = None, latency_ms: Optional[int] = None, namedScores: Optional[Any] = None):
        self.pass_ = pass_
        self.score = score
        self.reason = reason
        if component_results:
            self.component_results = [ComponentResult(**cr) for cr in component_results]
        if componentResults:
            self.componentResults = [ComponentResult(**cr) for cr in componentResults]
        if isinstance(assertion, Mapping):
            self.assertion = Assertion(**assertion)
        else:
            self.assertion = assertion
        self.latency_ms = latency_ms
        self.namedScores = namedScores

    
class GradingResult:
    pass_: bool
    score: float
    namedScores: Optional[Any]
    tokensUsed: Optional[TokenUsed]
    componentResults: Optional[List[ComponentResult]]
    assertion: Optional[Assertion]
    reason: Optional[str]

    def __init__(self, pass_: bool, score: float, namedScores: Optional[Any] = None, tokensUsed: Optional[TokenUsed] = None, componentResults: Optional[List[ComponentResult]] = None, assertion: Optional[Assertion] = None, reason : Optional[str] = None):
        self.pass_ = pass_
        self.score = score
    
        self.namedScores = namedScores
        if tokensUsed:
            self.tokensUsed = TokenUsed(**tokensUsed)
        if componentResults:
            self.componentResults = [ComponentResult(**cr) for cr in componentResults]
        if assertion:
            self.assertion = Assertion(**assertion)
        self.reason = reason


    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)




class Result:
    provider: Provider
    prompt: Prompt
    vars: Vars
    response: Response
    success: bool
    score: float
    namedScores: Any 
    latencyMs: int
    gradingResult: GradingResult
    error: Optional[str]

    def __init__(self, provider: Provider, prompt: Prompt, vars: Vars, response: Response, success: bool, score: float, namedScores: Any, latencyMs: int, gradingResult: GradingResult, error: Optional[str] = None):
        self.provider = Provider(**provider)
        self.prompt = Prompt(**prompt)
        self.vars = Vars(**vars)
        self.response = Response(**response)
        self.success = success
        self.score = score
        self.namedScores = namedScores
        self.latencyMs = latencyMs
        self.gradingResult = GradingResult(**gradingResult)
        self.error = error


class TokenUsage:
    total: int
    prompt: int
    completion: int
    cached: int

    def __init__(self, total: int, prompt: int, completion: int, cached: int):
        self.total = total
        self.prompt = prompt
        self.completion = completion
        self.cached = cached


class Stats:
    successes: int
    failures: int
    tokenUsage: TokenUsage

    def __init__(self, successes: int, failures: int, tokenUsage: TokenUsage):
        self.successes = successes
        self.failures = failures
        self.tokenUsage = TokenUsage(**tokenUsage)



class Metrics:
    score: float
    testPassCount: int
    testFailCount: int
    assertPassCount: int
    assertFailCount: int
    totalLatencyMs: int
    tokenUsage: TokenUsage
    namedScores: Any
    cost: int

    def __init__(self, score: float, testPassCount: int, testFailCount: int, assertPassCount: int, assertFailCount: int, totalLatencyMs: int, tokenUsage: TokenUsage, namedScores: Any, cost: int):
        self.score = score
        self.testPassCount = testPassCount
        self.testFailCount = testFailCount
        self.assertPassCount = assertPassCount
        self.assertFailCount = assertFailCount
        self.totalLatencyMs = totalLatencyMs
        self.tokenUsage = TokenUsage(**tokenUsage)
        self.namedScores = namedScores
        self.cost = cost

class TableHeadPrompt:
    raw: str
    display: str
    id: str
    provider: str
    metrics: Metrics

    def __init__(self, raw: str, display: str, id: str, provider:str, metrics: Metrics):
        self.raw = raw
        self.display = display
        self.id = id
        self.provider = provider
        self.metrics = Metrics(**metrics)

class Head:
    prompts: List[TableHeadPrompt]
    vars: Vars

    def __init__(self, prompts: List[TableHeadPrompt], vars: Vars):
        self.prompts = [TableHeadPrompt(**p) for p in prompts]
        self.vars = vars


class Output:
    pass_: bool
    score: float
    namedScores: Any
    text: str
    prompt: Prompt
    provider: str
    latencyMs: int
    gradingResult: GradingResult
    cost: int

    def __init__(self, pass_: bool, score: float, namedScores: Any, text: str, prompt: str, provider: str, latencyMs: int, gradingResult: GradingResult, cost: int):
        self.pass_ = pass_
        self.score = score
        self.namedScores = namedScores
        self.text = text
        self.prompt = prompt
        self.provider = provider
        self.latencyMs = latencyMs
        self.gradingResult = GradingResult(**gradingResult)
        self.cost = cost

class Assert:
    type: str
    value: str
    threshold: Optional[float]

    def __init__(self, type: str, value: str, threshold: Optional[float] = None):
        self.type = type
        self.value = value
        self.threshold = threshold

class Test:
    vars: Vars
    assert_: List[Assert]
    options: Any

    def __init__(self, vars: Vars, assert_: List[Assert], options: Any):
        self.vars = Vars(**vars)
        self.assert_ = [Assert(**a) for a in assert_]
        self.options = options


class Body:
    outputs: List[Output]
    test: Test
    vars: List[str]

    def __init__(self, outputs: List[Output], test: Test, vars: List[str]):
        self.outputs = [Output(**o) for o in outputs]
        self.test = Test(**test)
        self.vars = vars

class Table:
    head: Head
    body: List[Body]

    def __init__(self, head: Head, body: List[Body]):
        self.head = Head(**head)
        self.body = [Body(**b) for b in body]


class Results:
    version: int
    results: List[Result]
    stats: Stats
    table: Table

    def __init__(self, version: int, results: List[Result], stats: Stats, table: Table):
        self.version = version
        self.results = [Result(**r) for r in results]
        self.stats = Stats(**stats)
        self.table = Table(**table)

class ProviderConfig:
    id: str
    config: Any

    def __init__(self, id: str, config: Any):
        self.id = id
        self.config = config

class Config:
    prompts: List[str]
    providers: List[ProviderConfig]
    tests: List[str]
    sharing: bool
    outputPath: Optional[str]

    def __init__(self, prompts: List[str], providers: List[ProviderConfig], tests: List[str], sharing: bool, outputPath: Optional[str] = None):
        self.prompts = prompts
        self.providers = [ProviderConfig(**p) for p in providers]
        self.tests = tests
        self.sharing = sharing
        self.outputPath = outputPath

def load_from_json_file(path: str) -> Tuple[Results, Config]:
    with open(path) as f:
        json_str = f.read()

        # replace pass with pass_
        json_str = json_str.replace('"pass":', '"pass_":')
        json_str = json_str.replace('"assert":', '"assert_":')
        data = json.loads(json_str)
        results = Results(**data['results'])
        config = Config(**data['config'])

        return results, config