# Retrieval

This folder contains the scripts required to create vector databases and to test them.

## Create database
The [database_creator.py](./database_creator.py) script can be used to create a new database. 
Various settings can be made. Data can be prepared (removal of headers, ...) or loaded in its raw form. The extracted data is then divided into chunks and stored in a database. If the data is prepared, the parameters for the header and footer must be adapted to the document.

Several databases with different settings are currently created for each model:
- embedding_list = [ ... ]
- Chunk size: 1024, 2048, 4096
- Chunk overlap: 128, 256
- seperators: [‘\n\n’, ‘\n§’, ‘ §’]
- distance function: [‘cosine’, ‘l2’]
Depending on the application purpose, it may make sense to change these parameters. When changing them, however, make sure that the [custom-rag-loader-library](../libs/custom_rag_loader/) is kept up to date. We have used this to load the databases and LLMs.


All data is stored in a ChromaDB under the [databases/](./databases/) directory. 
> The databases must be created first after cloning, as the files would take up too much memory in git. (`python3 database_creator.py`).

ChromaDB can be accessed directly via its API, as well as via LanChain or LlamaIndex.


## Test database
The [tests](./tests) folder is used to test the databases. The tests are executed with Promptfoo. The directory contains three folders:
- [_input](./tests/_input/) - Input questions and script to create a formatted json file with the different test cases
- [_output](./tests/_output/) - Results of a test run. The [output.json](./tests/_output/output.json) contains the results of the last test run (As this file is very large, it should not be opened with an editor. In addition, it cannot be uploaded to github). To convert the data into a more usable format, the script [promptfoo_output_converter.py](./tests/_output/promptfoo_output_converter.py) can be used. This creates a compact json file with all the required information. To analyse the data and plot the results, the Jupyter notebook [plot_generation](./tests/_output/plot_generation.ipynb) can be used.
- rag](./tests/rag/) - This directory contains all the promptfoo configurations for the tests. The individual test cases are in the file [tests.yaml](./tests/rag/tests.yaml). The [promptfooconfig.yaml](./tests/rag/promptfooconfig.yaml) file contains all the settings for the various test settings.
It is important to ensure that all the configurations are entered once in yaml format for the test and that they are also transferred to the `label` in json format. This is required for later evaluation. To start a test run, the following command must be executed: `promptfoo eval -j 15 --no-cache`.