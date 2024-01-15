This is the python codebase for the "Signaux Faibles" project's failure prediction algorithms.

# Installation

## Cloning the repository

``` shell
git clone https://forge.dgfip.finances.rie.gouv.fr/dge/signaux-faibles/sf_datalake.git

```

## Prepare a virtual environment

The virtual environment allows one to install specific version of python packages independantly without messing with the system installation.

Create a python3 virtual environment

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

## Installing the package

### User

To install the package locally, run this command while inside the virtual environment

``` shell
pip install .
```

from the repository root.

### Developer

If you're contributing to the package development, you must install the package in editable mode and add some extra requirements for code testing: run this command while inside the virtual environment

``` shell
pip install -e .[test]
```

from the repository root.

Then, activate git hooks using

``` shell
pre-commit install --install-hooks -t pre-commit -t pre-push
```

This will install git hooks that should enforce a set of properties before committing / pushing code. These properties can be customized through the `pre-commit` config file and can cover a wide scope : coding style, code linting, tests, etc.

# Repository structure

- `.ci/` - Contains configuration associated with the maacdo API in order to execute jobs on a cluster using a CI pipeline. This is quite specific to the infrastructure used within the "signaux faibles" project.
- `docs/` - Sphinx auto-documentation sources.
- `src/` Contains all the python package source code, see the docs pages for a thorough description or the `__init__.py` module docstring.
    - `preprocessing/` - Production of datasets from raw data. Datasets loading and handling, exploration and feature engineering utilities.
    - `postprocessing/` - Generation of front-end ready documents, statistical evaluation of predictions.
- `tests/` - Tests associated with the code. They may be executed anytime using `pytest`.
- `.gitlab-ci.yml` - The gitlab CI/CD tools configuration file.
- `LICENSE` - The legal license associated with this repository.
- `MANIFEST.in` - Declaration of data resources used by the package.
- `.pre-commit-config.yaml` - Configuration file for the `pre-commit` package, responsible for pre-commit and pre-push git hooks.
- `.pylintrc` - Configuration file for the python linter.
- `pyproject.toml` and `setup.cfg` are configuration files for this package's setup.
- `README.md` - This file.
