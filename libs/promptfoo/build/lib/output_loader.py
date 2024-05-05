import json 

class Results:
    def __init__(self, version, results, stats, table):
        self.version = version
        self.results = [Result(**r) for r in results]
        self.stats = Stats(**stats)
        self.table = Table(**table)

class Result:
    def __init__(self, provider, prompt, vars, response, success, score, namedScores, latencyMs, gradingResult):
        self.provider = Provider(**provider)
        self.prompt = Prompt(**prompt)
        self.vars = Vars(**vars)
        self.response = Response(**response)
        self.success = success
        self.score = score
        self.namedScores = namedScores
        self.latencyMs = latencyMs
        self.gradingResult = GradingResult(**gradingResult)

class Provider:
    def __init__(self, id):
        self.id = id

class Prompt:
    def __init__(self, raw, display):
        self.raw = raw
        self.display = display

class Vars:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

class Response:
    def __init__(self, output):
        self.output = output

class GradingResult:
    def __init__(self, pass_, score, namedScores, tokensUsed, componentResults, assertion):
        self.pass_ = pass_
        self.score = score
        self.namedScores = namedScores
        self.tokensUsed = tokensUsed
        self.componentResults = [ComponentResult(**cr) for cr in componentResults]
        self.assertion = assertion

class ComponentResult:
    def __init__(self, pass_, score):
        self.pass_ = pass_
        self.score = score

class Stats:
    def __init__(self, successes, failures, tokenUsage):
        self.successes = successes
        self.failures = failures
        self.tokenUsage = tokenUsage

class Table:
    def __init__(self, head, body):
        self.head = Head(**head)
        self.body = [Body(**b) for b in body]

class Head:
    def __init__(self, prompts, vars):
        self.prompts = [Prompt(**p) for p in prompts]
        self.vars = vars

class Body:
    def __init__(self, outputs, test, vars):
        self.outputs = [Output(**o) for o in outputs]
        self.test = Test(**test)
        self.vars = vars

class Output:
    def __init__(self, pass_, score, namedScores, text, prompt, provider, latencyMs, gradingResult, cost):
        self.pass_ = pass_
        self.score = score
        self.namedScores = namedScores
        self.text = text
        self.prompt = prompt
        self.provider = provider
        self.latencyMs = latencyMs
        self.gradingResult = GradingResult(**gradingResult)
        self.cost = cost

class Test:
    def __init__(self, vars, assert_, options):
        self.vars = Vars(**vars)
        self.assert_ = [Assert(**a) for a in assert_]
        self.options = options

class Assert:
    def __init__(self, type, value):
        self.type = type
        self.value = value

def from_json(json_str) -> Results:
    data = json.loads(json_str)
    return Results(**data['results'])
