"""
"Signaux Faibles" project package for company failure prediction.

* ``config/`` - Configuration and model parameters that will be used during execution.
* ``__init__.py`` - Some data-related variables definitions.
* ``__main__.py`` - Main entry point script, which can be used to launch end-to-end
  predictions.
* ``evaluation.py`` - Model performance computations.
* ``explain.py`` - SHAP-based predictions explanation.
* ``exploration.py`` - Data exploration-dedicated functions.
* ``io.py`` - I/O functions.
* ``model_selection.py`` - Data sampling, model selection utilities.
* ``predictions.py`` - Post-process model predictions (generation of alert levels etc.)
* ``transform.py`` - Utilities and classes for handling and transforming datasets.
* ``utils.py`` - Misc utility functions (e.g. spark session handling, etc.)
"""
