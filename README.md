[![pipeline status](https://forge.dgfip.finances.rie.gouv.fr/raphaelventura/sf_datalake/badges/main/pipeline.svg)](https://forge.dgfip.finances.rie.gouv.fr/raphaelventura/sf_datalake/-/commits/main)

This is the python codebase for the "Signaux Faibles" project's failure prediction algorithms on the DGFiP-hosted datalake.

# Installation

## Cloning the repository

``` shell
git clone https://forge.dgfip.finances.rie.gouv.fr/raphaelventura/sf_datalake.git

```

## Prepare a virtual environment

The virtual environment allows one to install specific version of python packages independantly without messing with the system installation.

Create a virtual environment

``` shell
virtualenv -p `which python3` <virtualenv_dir>
```

Source the new virtual environment to begin working inside this environment

``` shell
source <virtualenv_dir>/bin/activate
```

Make sure the pip version packaged with the env is up to date (it should be >= 19)

``` shell
pip install -U pip
```

Install the sf-datalake package inside the environment

``` shell
pip install .
```

from the repository root.

## Activate git hooks

Activate git hooks using

``` shell
pre-commit install
```

This will install git hooks that should enforce a set of properties before committing / pushing code. These properties can be customized through the `pre-commit` config file and can cover a wide scope : coding style, code linting, tests, etc.

# Repository structure

- `.ci/` - Contains configuration associated with the maacdo API in order to execute jobs on the datalake using a CI pipeline.
- `datasets_handling/` - Production of datasets from raw data. Datasets loading and handling, exploration and feature engineering utilities.
- `docs/` - Sphinx auto-documentation sources (see `datascience_workflow.md`) and textual / tabular documentation of the data used for training and prediction.
- `notebooks/` - Jupyter notebooks that leverage the package code. These may typically be used for tutorials / presentations.
- `src/` Contains all the package code:
    - `config/` - Configuration and model parameters that will be used during execution.
    - `processing/` - Data processing and models execution.
    - `__init__.py` - Some data-related variables definitions.
    - `__main__.py` - Main entry point script, which can be used to launch end-to-end predictions.
    - `evaluation.py` - Scores computations.
    - `exploration.py `- Data exploration-dedicated functions.
    - `io.py` - I/O functions.
    - `model.py` - Model utilities and classes.
    - `sampler.py` - Data sampling functions.
    - `transform.py` - Utilities and classes for handling and transforming datasets.
    - `utils.py` - Utility functions for spark session and data handling.
- `test/` - Tests (unitary, integration) associated with the code. They may be executed anytime using `pytest`.
- `datalake DGFiP.md` - Info about handing jobs over to the datalake and use of the jupyter lab.
- `datascience_workflow.md`- describes the workflow for data scientists working on the project.
- `.gitlab-ci.yml` - The gitlab CI/CD tools configuration file.
- `LICENSE` - The legal license associated with this repository.
- `MANIFEST.in` - Declaration of data resources used by the package.
- `.pre-commit-config.yaml` - Configuration file for the `pre-commit` package, responsible for pre-commit and pre-push git hooks.
- `.pylintrc` - Configuration file for the python linter.
- `pyproject.toml` and `setup.cfg` are configuration files for this package's setup.
- `README.md` - This file.
- `setup.cfg` - Configuration file for this package's setup by pip.

# Documentation

Documentation can be generated by executing

``` shell
make html
```

from the `docs/` folder. This will produce a directory containing an html-formatted documentation "readthedocs-style". This doc can be browsed by opening `docs/build_/html/index.html`.

Other formats are available for export (e.g., pdf, man, texinfo); for more info, execute

``` shell
make help
```

from the `docs/` folder as well.

Documentation is generated based on the `.rst` files contained inside `docs/source`. If needed, these files can be automatically generated using `sphinx`: from the `docs/` repository, execute

``` shell
sphinx-apidoc -fP -o source/ ../sf_datalake
```
