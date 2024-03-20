"""Explain AI predictions using shap package.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyspark.ml.classification
import shap

import sf_datalake.transform
import sf_datalake.utils


def explanation_data(
    features_list: List[str],
    features_column: str,
    model: pyspark.ml.Model,
    train_data: pyspark.sql.DataFrame,
    prediction_data: pyspark.sql.DataFrame,
    n_train_sample: int,
) -> Tuple[pd.DataFrame, float]:
    # pylint:disable=too-many-arguments
    """Compute Shapeley coefficients + expected value for predictions.

    Shapeley coefficients represent the contribution to a model output. The computed
    units depend on the explained model. For instance, for logistic regression, and
    gradient-boosted trees, the coefficients computed by shap are in log-odds units.

    Args:
        features_list: A list of names of features, sorted as they were inserted into
          `features_column`.
        features_column: A column containing the model's features.
        model: A pyspark model used for prediction.
        train_data: Training dataset.
        prediction_data: Prediction dataset.
        n_train_sample: Number of training set samples used for estimating features
          correlation.

    Returns:
        A tuple containing:
        - A pandas DataFrame containing shap values associated with each feature.
        - The expected failure probability value over the prediction dataset.

    """
    X_prediction = sf_datalake.transform.vector_disassembler(
        df=prediction_data,
        columns=features_list,
        assembled_col=features_column,
        keep=["siren"],
    ).toPandas()

    if isinstance(model, pyspark.ml.classification.LogisticRegressionModel):
        assert n_train_sample > 0 and isinstance(
            n_train_sample, int
        ), "n_train_sample must be a positive integer."
        X_train_sample = (
            sf_datalake.transform.vector_disassembler(
                df=train_data, columns=features_list, assembled_col=features_column
            )
            .sample(fraction=min(1.0, n_train_sample / train_data.count()))
            .toPandas()
        )

        explainer = shap.LinearExplainer(
            model=(model.coefficients.toArray(), model.intercept),
            nsamples=n_train_sample,
            data=X_train_sample,
            feature_perturbation="correlation_dependent",
        )
        shap_values = explainer.shap_values(X_prediction[features_list].values)

    elif isinstance(
        model,
        (
            pyspark.ml.classification.DecisionTreeClassificationModel,
            pyspark.ml.classification.RandomForestClassificationModel,
            pyspark.ml.classification.GBTClassificationModel,
        ),
    ):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(
            X_prediction[features_list].values, check_additivity=False
        )
    else:
        raise NotImplementedError(f"{model} models are not supported.")

    # Here tree-based models may output a list of two elements corresponding to the
    # two (complementary) classes. Weirdly enough, this seems to happen only with
    # random forest or decision tree classifiers… Hence the `[1]` indexing.
    if isinstance(
        model,
        (
            pyspark.ml.classification.RandomForestClassificationModel,
            pyspark.ml.classification.DecisionTreeClassificationModel,
        ),
    ):
        shap_values = shap_values[1]
        explainer.expected_value = explainer.expected_value[1]

    sv = pd.DataFrame(
        shap_values,
        index=X_prediction["siren"],
    )
    ev = explainer.expected_value

    sv.columns = features_list
    return sv, ev


def explanation_scores(
    shap_df: pd.DataFrame,
    topic_groups: Dict[str, List[str]],
    n_concerning: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute plot-ready feature contribution.

    This computes individual, as well as aggregated, features contributions. The most
    significant contributions (in favor of a positive prediction) are returned as a
    DataFrame containing concerning feature names and values.

    Contributions are first summed within feature groups:
    - at a "feature" scale: lagged variables and such, for a given feature.
    - at a "topic" scale: features that describe information within a common topic.
      The corresponding groups will be used as main axes for visualisation.

    Args:
        shap_df: The shap values associated with the features used for machine learning.
        topic_groups: A grouping of features, by major topic.
        n_concerning: Number of most significant features to return.

    Returns:
        A 2-uple containing:
        - A "macro scores" df, which contains aggregated features contrbutions across a
          topic group.
        - A "concerning scores" df, which contains the most significant individual
          features contributions.

    """
    feature_groups: Dict[str, str] = {}
    macro_features = set(
        feature for flist in topic_groups.values() for feature in flist
    )
    feature_lvl_df = pd.DataFrame(index=shap_df.index)
    macro_scores = pd.DataFrame(index=shap_df.index)

    # Get feature-level shap values
    for feature in shap_df.columns:
        source_variable_found = False
        for mfeature in macro_features:
            if feature.startswith(mfeature):
                feature_groups.setdefault(mfeature, []).append(feature)
                source_variable_found = True
                break
        if not source_variable_found:
            raise ValueError(f"Could not find source variable for feature {feature}.")
    for group, features in feature_groups.items():
        feature_lvl_df[group] = shap_df[features].sum(axis=1)

    # Compute 'macro' scores per topic group
    for group, features in topic_groups.items():
        macro_scores.loc[:, f"{group}_macro_score"] = feature_lvl_df[features].sum(
            axis=1
        )

    # Concerning features (with the most significant feature-level shap values)
    sorter = np.argsort(-feature_lvl_df.values, axis=1)[:, :n_concerning]
    concerning_feat = pd.DataFrame(
        feature_lvl_df.columns[sorter], index=feature_lvl_df.index
    )
    concerning_values = pd.DataFrame(
        feature_lvl_df.values[np.arange(len(feature_lvl_df))[:, np.newaxis], sorter],
        index=feature_lvl_df.index,
    )
    concerning_feat.columns = [f"concerning_feat_{n}" for n in range(n_concerning)]
    concerning_values.columns = [f"concerning_val_{n}" for n in range(n_concerning)]

    return macro_scores, concerning_feat.join(concerning_values)
