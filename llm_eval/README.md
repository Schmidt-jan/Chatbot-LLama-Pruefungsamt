# Evaluation Guide

This document outlines the process for conducting test runs and analyzing the results within your project.

## Directory and File Structure

The following files and directories are crucial components of the project:

- **`/input/test_data.yaml`**: Contains the test questions and reference answers that serve as the basis for the test runs. Additionally, this file allows you to define the metrics that will be used to evaluate the test results.
- **`/llm_eval_run/run_promptfoo.sh`**: A Bash script that executes the test runs, converts the test cases into the appropriate Promptfoo format, and initiates the evaluation process.
- **`/metrics`**: A directory containing the metrics used for evaluating the test results.
- **`/output/promptfoo_data`**: Stores the results of the test runs in both processed and raw formats.
- **`/output/plots`**: This directory holds the generated plots that visualize the test results.
- **`/promptfooconfig.yaml`**: A configuration file where you can adjust the LLM models (providers) used in the tests.
- **`/llmrun_eval_plotter.py`**: A Python script responsible for generating the plots and visualizing the evaluation results. Within this script, you can also specify the desired test runs for combined analysis through the `promptfoo_runs` variable.
y
## Initiate Testruns


### Step 1: Defining the Test Runs

The test runs are defined in the file [`/input/test_data.json`](./input/test_data.json). This file includes:

- **Test Questions and Reference Answers**: These form the basis for the test runs.
- **Metrics**: The metrics to be used for evaluating the test results can also be configured in this file. Adjust these metrics according to the specific requirements of your analysis.

#### Adjusting the LLM Models

The file [`/promptfooconfig.yaml`](./promptfooconfig.yaml) allows you to configure the LLM models used in the tests. This configuration file enables you to select the desired provider and adjust the models to meet the specific needs of your tests.


### Step 2: Executing the Test Runs

Once the test runs are defined, execute the Bash script [`/llm_eval_run/run_promptfoo.sh`](./llm_eval_run/run_promptfoo.sh). This script performs the following tasks:

```console
cd /Chatbot-LLama-Pruefungsamt/llm_eval/llm_eval_run
./run_promptfoo.sh
```

1. **Conversion**: The defined test cases are converted into the appropriate Promptfoo format.
2. **Evaluation**: The tests are executed using Promptfoo.
3. **Result Storage**: The results are stored in both processed and raw formats in the directory [`/output/promptfoo_data`](./output/promptfoo_data).

#### Note on Server Load

Please be aware that running the LLM models may significantly increase server load. Ensure that the server is adequately managed to avoid performance issues during test execution.

### Step 3: Analyzing the Results

During the test runs, the results are evaluated using the metrics defined in the directory [`/metrics`](./metrics). Upon completion of the test runs, the Bash script automatically initiates the process for analyzing and plotting the results.

#### Plotting and Visualization

The analysis generates a variety of plots to visualize the test results. These plots are stored in the directory [`/output/plots`](./output/plots).

The Python script [`/llmrun_eval_plotter.py`](./llmrun_eval_plotter.py) is responsible for creating the plots. Additionally, you can specify the test runs to be included in the combined analysis by configuring the `promptfoo_runs` variable within this script.

## Notes

- The plots provide valuable insights into the performance of the tested models and should be carefully analyzed.
- Monitor the server load during the execution of the LLMs to prevent potential performance issues.

## Summary

This guide enables you to define test runs, execute them, and comprehensively analyze the results. The combination of automated testing and subsequent visualization facilitates an effective evaluation of the models under test.
