# picasso-workflow

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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

- Dependencies are specified in requirements.txt
- Separately install picassosr (left out of requirements.txt for flexibility
of installing via pip or a development version)
- Should be platform independent. Tested on MacOS Sonoma and  Windows Server.

## Usage

- see examples in the folder "examples".

## Contributing

- For adding new workflow modules, create a new branch (feature/newmodule),
and add new modules to:
	- util/AbstractModuleCollection
	- analyse/AutoPicasso
	- confluence/ConfluenceReporter
- Please adhere to PEP code style and send pull request when done.

## License

This project is licensed under the [MIT License](LICENSE).
