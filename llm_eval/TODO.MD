# ToDo's for LLM EVAL


- [ ] Change absolute paths to relative (e.g. [output_parser.py](output/output_parser.py),  [test_data_JsonToYaml.py](input/test_data_JsonToYaml.py)) or have them be detected automatically, same for [custom_rag_loader.py](../libs/custom_rag_loader/custom_rag_loader.py) (maybe via an env variable?)
- [ x ] Fix keywords array in vars of [test.yaml](input/tests.yaml), promptfoo runs tests for each possible combination in the keywords array (see https://www.promptfoo.dev/docs/configuration/guide/#multiple-variables-in-a-single-test-case)
- [  ] Maybe automate visualization and evaluation of generated output (atleast centralize it)