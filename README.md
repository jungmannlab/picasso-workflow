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

## Contributing

- For adding new workflow modules, create a new branch (feature/newmodule),
and add new modules to:
	- util/AbstractModuleCollection
	- analyse/AutoPicasso
	- confluence/ConfluenceReporter
	- tests/test_analyse
	- tests/test_confluence
- Please adhere to PEP code style and send pull request when done.

## License

This project is licensed under the [MIT License](LICENSE).
