# picasso-workflow

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

master:
![master test](https://img.shields.io/github/actions/workflow/status/jungmannlab/picasso-workflow/run-unittests.yml?branch=master)

develop:
![develop test](https://img.shields.io/github/actions/workflow/status/jungmannlab/picasso-workflow/run-unittests.yml?branch=develop)
![Coveralls develop](https://img.shields.io/coverallsCoverage/github/jungmannlab/picasso-workflow?branch=develop)


A package for automated DNA-PAINT analysis workflows

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- The project aims at automating DNA-PAINT workflows, especially the analysis
via picassosr.
- There are two main types of workflow:
	- Single-dataset workflow: a single dataset is e.g. loaded, localized,
	and clustered.
	- Aggregation workflow: multiple datasets undergo a single-dataset
	workflow and are then aggregated.

## Installation

### Prerequisites

Make sure to have (ana)conda installed. On Mac OS, open the terminal (command + space,
type "terminal" hit enter). Then, one after another execute the follwing commands
- `curl -O https://repo.anaconda.com/archive/Anaconda3-2024.09-MacOSX-x86_64.sh`
- `bash Anaconda3-2024.09-MacOSX-x86_64.sh`
- `~/anaconda3/bin/conda init`
- `conda config --remove channels defaults`
- `conda config --add channels conda-forge`
- close the terminal and reopen it, to apply the changes.

### picasso-workflow specific installation

- create a new anaconda environment: `conda create -n picasso-workflow python=3.10`
- If you want to use a local development version of picasso, install that first:
	- `cd /path/to/picasso`
	- `pip install -r requirements.txt`
- Dependencies are specified in requirements.txt, install by:
	- `cd /path/to/picasso-workflow`
	- `pip install -e .`
- Should be platform independent. Tested on MacOS Sonoma and  Windows Server.

## Usage

- see examples in the folder "examples".
- if you have access, see examples in "/Volumes/pool-miblab/users/grabmayr/picasso-workflow_testdata"

## Contributing

- Install pre commit hooks:
	- `pip install pre-commit` (if not already installed by requirements in pyproject.toml / pip install -e)
	- `cd GitHub/picasso-workflow`
	- `pre-commit install`
	- Now, before commit via git, the hooks will run through and check code and style
	- optionally, the hooks can be run manually: `pre-commit run --all-files`
- For adding new workflow modules, create a new branch (feature/newmodule),
and add new modules as methods to the following classes:
	- util/AbstractModuleCollection: This 'registers' the module as an option provided to the user. It does not need to do anything (pass).
	- analyse/AutoPicasso: The code here actually does the analysis work. Optimally, only data cleanup and reshaping (unit conversions etc) are done here, and functions of picasso are called. As an intermediate solution, self-sustaining functions that would fit into picasso scope-wise can be put into the module `picasso_outpost` and called from here. Refer to other modules as blueprints for how to use the `parameters` and `results` arguments. Modules for use in Single-dataset analys stage make use of recarray `self.locs` and list `self.info` as provided by picasso, while modules for use in aggregation stage make use of the list of recarrays `self.channel_locs` and list of list of dicts `self.channel_info` as well as the list of strings `self.channel_tags`, to accommodate localizations and information, as well as the names given to the various datasets to analyse (resulting from the single dataset stages).
	- confluence/ConfluenceReporter: Make sure to generate meaningful output to Confluence describing the results of your module, preferably including figures. Refer to other modules as blueprints for how to do that.
	- tests/test_analyse/TestAnalyseModules: make sure to write an unittest for your module in AutoPicasso. It should test as many lines of code of your module as possible (coverage), and make sure they run through. The meaningfulness of the output can be tested too, but there's no need to overdo it, as this might require you to simulate meaningful input data. The latter should definitely be tested by using real, acquired data in a complete workflow (integration test).
	- tests/test_confluence/Test_B_ConfluenceReporterModules: As above, write a unit test for your module to make sure it runs through.
 	- tests/test_picasso_outpost: In case you added functions here, make sure to test them as well.
  - for running the unittests, go to terminal, activate the conda environment by `conda activate picasso-workflow`, navigate to your git repository, e.g. `cd ~/GitHub/picasso-workflow`, and enter `pytest -v`. If you don't want to run all tests for efficiency, you can add the module as an argument
- Please document your code. Importantly, write docstrings in the beginning of your function, in Google format (https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google)
- Please adhere to PEP code style (https://peps.python.org/pep-0008/) and send pull request when done.
- Please be aware: Pre-commit hooks have been added to ascertain that the pushed code is clean and readable.
- GitHub Actions are in place for the master and develop branches, running unittests and analysing coverage at new pushes [currently under construction]

## License

This project is licensed under the [MIT License](LICENSE).
