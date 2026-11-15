"""Inherently interpretable model candidates.

Extend or replace :func:`get_models` with the models you want to compare, e.g. trees of different
depths to probe the threshold at which a model stops being interpretable, or random forests and
gradient-boosted trees to cover a whole model class.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    return {
        "Tree (Depth 2)": DecisionTreeClassifier(max_depth=2, random_state=seed),
        "Tree (Depth 5)": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "Tree (Unrestricted)": DecisionTreeClassifier(max_depth=None, random_state=seed),
        "Forest (3 Trees)": RandomForestClassifier(n_estimators=3, max_depth=3, random_state=seed),
        "Forest (10 Trees)": RandomForestClassifier(n_estimators=10, max_depth=5, random_state=seed),
        "Forest (50 Trees)": RandomForestClassifier(n_estimators=50, max_depth=None, random_state=seed)
    }