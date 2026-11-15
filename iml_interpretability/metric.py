"""Interpretability metrics.

Implement your own metric by subclassing :class:`InterpretabilityMetric`. The metric must be
applicable to every model of the chosen model class (e.g. decision trees, random forests and
gradient-boosted trees), so that scores are comparable across models and datasets.
:class:`RelativeNumberOfLeaves` is a minimal example for a single decision tree.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


class InterpretabilityComplexityScore(InterpretabilityMetric):
    """
    Calculates an Interpretability Complexity Score (ICS).
    Lower scores indicate higher interpretability (easier for humans to understand).
    
    The score combines:
    1. Size of the model (Total Nodes)
    2. Cognitive depth (Max Depth)
    3. Feature sparsity (Unique Features Used)
    
    For ensembles, the complexity is the sum of the complexity of all individual trees.
    """
    
    name = "ics_score"

    def __init__(self, alpha=1.0, beta=2.0, gamma=5.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _score_single_tree(self, tree) -> float:
        """Helper to extract metrics from the underlying C tree structure."""
        total_nodes = tree.node_count
        # feature array contains feature indices, -2 indicates a leaf node
        used_features = set([f for f in tree.feature if f != -2])
        num_unique_features = len(used_features)
        max_depth = tree.max_depth
        
        return (self.alpha * total_nodes) + (self.beta * max_depth) + (self.gamma * num_unique_features)

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the ICS interpretability score for trees and ensembles."""
        
        # Handle ensembles like RandomForest and GradientBoosting
        if hasattr(model, 'estimators_'):
            # Flatten in case of GradientBoosting which stores estimators in a 2D array
            estimators = np.array(model.estimators_).flatten()
            scores = []
            for est in estimators:
                # Retrieve the underlying tree structure
                tree_struct = est.tree_ if hasattr(est, 'tree_') else est
                scores.append(self._score_single_tree(tree_struct))
            return float(np.sum(scores))
            
        # Handle single Decision Trees
        elif hasattr(model, 'tree_'):
            return float(self._score_single_tree(model.tree_))
            
        else:
            raise TypeError(f"{type(self).__name__} only supports tree-based models, got {type(model).__name__}")