from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import cross_val_score


class State:
    """
    Represents a feature-subset state in the search graph.

    Attributes:
        number: [depth, index] where depth == number of selected features
        description: list of selected feature indices
        v_value: TD value estimate for this state
        reward: cached evaluation score for this subset
        nb_visited: number of visits to this state
    """

    def __init__(
        self,
        number: list,
        description: list,
        v_value: float = 0.0,
        reward: Optional[float] = None,
        nb_visited: int = 0,
    ) -> None:
        self.number = number
        self.description = sorted(description)
        self.v_value = float(v_value)
        self.reward = reward
        self.nb_visited = int(nb_visited)

    @property
    def key(self) -> frozenset:
        return frozenset(self.description)

    def get_reward(
        self,
        clf,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring_classifier: str = "accuracy",
        scoring_regressor: str = "r2",
        deduplicate: bool = True,
        n_jobs: int = 1,
    ) -> float:
        """
        Evaluate and cache the state reward using cross-validation.

        In the referenced papers, this reward is the model assessment value V(F).
        """
        if self.reward is not None:
            return float(self.reward)

        if not self.description:
            self.reward = 0.0
            return self.reward

        df = pd.concat([X.iloc[:, self.description], y], axis=1)
        if deduplicate:
            df = df.drop_duplicates(ignore_index=True)

        X_subset = df.iloc[:, :-1]
        y_subset = df.iloc[:, -1]

        effective_cv = self._resolve_cv(clf, y_subset, cv)
        if effective_cv < 2:
            self.reward = 0.0
            return self.reward

        if is_classifier(clf):
            scoring = scoring_classifier
        elif is_regressor(clf):
            scoring = scoring_regressor
        else:
            raise ValueError("Unsupported estimator type: expected classifier or regressor")

        scores = cross_val_score(clf, X_subset, y_subset, cv=effective_cv, scoring=scoring, n_jobs=n_jobs)
        self.reward = float(np.mean(scores))
        return self.reward

    @staticmethod
    def _resolve_cv(clf, y: pd.Series, requested_cv: int) -> int:
        """
        Keep CV feasible for small datasets/classes.
        """
        if requested_cv < 2:
            return 2

        if is_classifier(clf):
            class_counts = np.array(y.value_counts())
            if class_counts.size == 0:
                return 0
            return int(min(requested_cv, np.min(class_counts)))

        return int(min(requested_cv, len(y)))

    def update_v_value(self, alpha: float, gamma: float, reward_delta: float, next_state: "State") -> float:
        """
        TD(0) update:
            V(F_t) <- V(F_t) + alpha * (r_t + gamma * V(F_{t+1}) - V(F_t))
        where r_t = V(F_{t+1}) - V(F_t).
        """
        td_target = reward_delta + gamma * next_state.v_value
        self.v_value = self.v_value + alpha * (td_target - self.v_value)
        return self.v_value

    def is_final(self, nb_of_features: int) -> bool:
        return len(self.description) >= nb_of_features

    def is_equal(self, compared_state: "State") -> bool:
        return set(self.description) == set(compared_state.description)
