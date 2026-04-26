from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVC

from .action import Action
from .fsrlearning import FeatureSelectionProcess
from .state import State


class FeatureSelectorRL:
    """
    RL-based feature selector that follows the TD/AOR strategy from:
    - FSTD (2013)
    - the 2021 extension with random episode starts and epsilon-greedy bandit phase
    """

    def __init__(
        self,
        feature_number: int,
        nb_explored: list | None = None,
        nb_not_explored: list | None = None,
        feature_structure: dict | None = None,
        aor: list | None = None,
        eps: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 0.70,
        nb_iter: int = 100,
        explored: int = 0,
        not_explored: int = 0,
        starting_state: str = "empty",
        strategy: str = "2021",
        exploration_c: float = 1.0,
        exploit_b: float = 0.5,
        max_consecutive_declines: Optional[int] = None,
        cv: int = 5,
        n_jobs: int = 1,
        scoring_classifier: str = "accuracy",
        scoring_regressor: str = "r2",
        random_state: int | None = None,
    ):
        self.feature_number = int(feature_number)
        self.nb_explored = nb_explored
        self.nb_not_explored = nb_not_explored
        self.feature_structure = feature_structure
        self.aor = aor

        self.eps = float(eps)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.nb_iter = int(nb_iter)

        self.explored = int(explored)
        self.not_explored = int(not_explored)

        self.starting_state = starting_state
        self.strategy = strategy
        self.exploration_c = float(exploration_c)
        self.exploit_b = float(exploit_b)
        self.max_consecutive_declines = (
            int(max_consecutive_declines)
            if max_consecutive_declines is not None
            else int(max(1, round(np.sqrt(self.feature_number))))
        )

        self.cv = int(cv)
        self.n_jobs = int(n_jobs)
        self.scoring_classifier = scoring_classifier
        self.scoring_regressor = scoring_regressor
        self.random_state = random_state

    def fit_predict(self, X, y, clf=None) -> tuple[list, int]:
        """
        Train and return feature ranking artifacts.

        Return:
            ([feature_index, n_selected, aor_values, ranking], visited_state_count)
        """
        self.aor = [np.zeros(self.feature_number, dtype=float), np.zeros(self.feature_number, dtype=float)]
        self.feature_structure = {}
        self.nb_explored = []
        self.nb_not_explored = []
        self.explored = 0
        self.not_explored = 0

        process = FeatureSelectionProcess(
            self.feature_number,
            self.eps,
            self.alpha,
            self.gamma,
            self.aor,
            self.feature_structure,
            random_state=self.random_state,
        )

        X = pd.DataFrame(X)
        y = pd.Series(y)

        if clf is None:
            clf = SVC(kernel="rbf", gamma="scale")

        for _ in tqdm(range(self.nb_iter)):
            if self.starting_state == "random":
                current_state, _ = process.pick_random_state()
            elif self.starting_state == "empty":
                current_state, _ = process.start_from_empty_set()
            else:
                raise ValueError("starting_state must be either 'empty' or 'random'")

            current_state.get_reward(
                clf,
                X,
                y,
                cv=self.cv,
                scoring_classifier=self.scoring_classifier,
                scoring_regressor=self.scoring_regressor,
                n_jobs=self.n_jobs,
            )

            consecutive_declines = 0

            while True:
                if current_state.is_final(self.feature_number):
                    process.add_to_historic(current_state)
                    break

                available_features = process.get_available_features(current_state)
                if not available_features:
                    process.add_to_historic(current_state)
                    break

                first_visit = current_state.nb_visited == 0
                selected_feature, used_exploration = self._select_feature(
                    current_state=current_state,
                    available_features=available_features,
                    first_visit=first_visit,
                    process=process,
                )

                next_state = process.get_or_create_state(current_state.description + [selected_feature])
                next_state.get_reward(
                    clf,
                    X,
                    y,
                    cv=self.cv,
                    scoring_classifier=self.scoring_classifier,
                    scoring_regressor=self.scoring_regressor,
                    n_jobs=self.n_jobs,
                )

                reward_delta = float(next_state.reward) - float(current_state.reward)
                current_state.update_v_value(self.alpha, self.gamma, reward_delta, next_state)

                action = Action(current_state, next_state)
                process.aor = action.get_aorf(process.aor, reward_delta=reward_delta)
                process.update_state_action_repository(current_state, selected_feature, reward_delta)

                process.add_to_historic(current_state)

                if used_exploration:
                    self.explored += 1
                else:
                    self.not_explored += 1

                if reward_delta < 0:
                    consecutive_declines += 1
                else:
                    consecutive_declines = 0

                current_state = next_state

                if consecutive_declines >= self.max_consecutive_declines:
                    process.add_to_historic(current_state)
                    break

            self.nb_explored.append(self.explored)
            self.nb_not_explored.append(self.not_explored)

        self.aor = process.aor
        self.feature_structure = process.feature_structure

        results = process.get_final_aor_sorted()
        visited_total = self.nb_explored[-1] + self.nb_not_explored[-1]
        return results, visited_total

    def _select_feature(
        self,
        current_state: State,
        available_features: list[int],
        first_visit: bool,
        process: FeatureSelectionProcess,
    ) -> tuple[int, bool]:
        rng = process.rng

        if first_visit:
            return int(rng.choice(available_features)), True

        if bool(rng.binomial(1, self.eps)):
            return int(rng.choice(available_features)), True

        if self.strategy == "2013":
            return self._exploit_2013(current_state, available_features, process), False
        if self.strategy == "2021":
            return self._exploit_2021(current_state, available_features, process), False

        raise ValueError("strategy must be '2013' or '2021'")

    def _exploit_2021(
        self,
        current_state: State,
        available_features: list[int],
        process: FeatureSelectionProcess,
    ) -> int:
        """
        Bandit-based exploit phase (UCB-style).
        """
        repo = process.state_action_repository[current_state.key]
        t_state = max(1, int(sum(int(stat["count"]) for stat in repo.values())))

        scores = []
        for feature in available_features:
            if feature in repo:
                local_mean = float(repo[feature]["mean"])
                n_feature = float(repo[feature]["count"])
            else:
                # fallback to global AOR for unseen actions in this state
                local_mean = float(process.aor[1][feature])
                n_feature = 0.0

            bonus = self.exploration_c * np.sqrt(np.log(t_state + 1.0) / (n_feature + 1.0))
            scores.append(local_mean + bonus)

        return self._argmax_with_random_tie_break(scores, available_features, process.rng)

    def _exploit_2013(
        self,
        current_state: State,
        available_features: list[int],
        process: FeatureSelectionProcess,
    ) -> int:
        """
        FSTD exploit function:
        - if round(T_F^b) == round((T_F+1)^b): pick from repository via UCB
        - else: pick a new feature by max AOR
        """
        repo = process.state_action_repository[current_state.key]
        t_state = max(1, current_state.nb_visited)

        use_repository = round(t_state ** self.exploit_b) == round((t_state + 1) ** self.exploit_b)

        if use_repository:
            candidate_repo_features = [f for f in available_features if f in repo]
            if candidate_repo_features:
                scores = []
                total_count = max(1, int(sum(int(repo[f]["count"]) for f in candidate_repo_features)))
                for feature in candidate_repo_features:
                    mean_reward = float(repo[feature]["mean"])
                    count = float(repo[feature]["count"])
                    bonus = self.exploration_c * np.sqrt(np.log(total_count + 1.0) / count)
                    scores.append(mean_reward + bonus)
                return self._argmax_with_random_tie_break(scores, candidate_repo_features, process.rng)

        # New feature according to max global AOR.
        unseen_here = [f for f in available_features if f not in repo]
        candidates = unseen_here if unseen_here else available_features
        scores = [float(process.aor[1][f]) for f in candidates]
        return self._argmax_with_random_tie_break(scores, candidates, process.rng)

    @staticmethod
    def _argmax_with_random_tie_break(scores: list[float], features: list[int], rng) -> int:
        arr = np.asarray(scores, dtype=float)
        max_score = np.max(arr)
        ties = np.where(np.isclose(arr, max_score))[0]
        pick = int(rng.choice(ties))
        return int(features[pick])

    def get_plot_ratio_exploration(self):
        plt.plot(
            [i for i in range(len(self.nb_not_explored))],
            self.nb_not_explored,
            label="Exploit selections",
        )
        plt.plot(
            [i for i in range(len(self.nb_explored))],
            self.nb_explored,
            label="Explore selections",
        )
        plt.xlabel("Number of iterations")
        plt.ylabel("Number of selections")
        plt.legend(loc="upper left")
        plt.show()

    def get_feature_strengh(self, results):
        res = results[0] if isinstance(results, tuple) else results
        indices = list(res[0])
        values = list(res[2])

        plt.bar(x=indices, height=values, color=["blue" if v >= 0 else "red" for v in values])
        plt.xlabel("Feature index")
        plt.ylabel("Average feature reward (AOR)")
        plt.show()

    def get_depth_of_visited_states(self):
        sum_depth = []
        for depth in sorted(self.feature_structure.keys()):
            sum_index = [st.nb_visited for st in self.feature_structure[depth]]
            sum_depth.append(np.sum(sum_index))

        plt.plot([i for i in range(len(sum_depth))], sum_depth)
        plt.xlabel("Depth of visited states")
        plt.ylabel("Number of visits")
        plt.plot()

    def compare_with_benchmark(self, X, y, results) -> list:
        """
        Compare RL ranking against RFE baseline over subset sizes.
        """
        is_better_list: list = []
        avg_benchmark_accuracy: list = []
        avg_rl_accuracy: list = []

        rl_res = results[0] if isinstance(results, tuple) else results

        X = pd.DataFrame(X)
        y = pd.Series(y)

        for i in range(1, self.feature_number):
            # RL-based subset
            clf = RandomForestClassifier(n_jobs=self.n_jobs, random_state=self.random_state)
            df = pd.concat([X.iloc[:, rl_res[-1][i:]], y], axis=1)
            df = df.drop_duplicates(ignore_index=True)

            min_samples = np.min(np.array(df.iloc[:, -1].value_counts()))
            if min_samples < 5 and min_samples >= 2:
                accuracy = np.mean(
                    cross_val_score(
                        clf,
                        df.iloc[:, :-1],
                        df.iloc[:, -1],
                        cv=min_samples,
                        scoring="balanced_accuracy",
                        n_jobs=self.n_jobs,
                    )
                )
            elif min_samples < 2:
                accuracy = 0.0
            else:
                accuracy = np.mean(
                    cross_val_score(
                        clf,
                        df.iloc[:, :-1],
                        df.iloc[:, -1],
                        cv=5,
                        scoring="balanced_accuracy",
                        n_jobs=self.n_jobs,
                    )
                )

            # Benchmark RFE subset
            estimator = RandomForestClassifier(n_jobs=self.n_jobs, random_state=self.random_state)
            selector = RFE(estimator, n_features_to_select=len(rl_res[-1]) - i, step=1)
            cv_results = cross_validate(selector, X, y, cv=5, scoring="balanced_accuracy", return_estimator=True)
            sele_acc = np.mean(cv_results["test_score"])

            is_better_list.append(1 if accuracy >= sele_acc else 0)
            avg_benchmark_accuracy.append(sele_acc)
            avg_rl_accuracy.append(accuracy)

        index_variable = [i for i in range(len(avg_benchmark_accuracy))]

        avg_benchmark_accuracy.reverse()
        avg_rl_accuracy.reverse()

        if len(index_variable) >= 4:
            avg_benchmark_smooth = make_interp_spline(index_variable, avg_benchmark_accuracy)
            avg_rl_smooth = make_interp_spline(index_variable, avg_rl_accuracy)
            x_axis = np.linspace(np.min(index_variable), np.max(index_variable), 100)
            y_benchmark = avg_benchmark_smooth(x_axis)
            y_rl = avg_rl_smooth(x_axis)
            plt.plot(x_axis, y_benchmark, label="Benchmark accuracy")
            plt.plot(x_axis, y_rl, label="RL accuracy")
        else:
            plt.plot(index_variable, avg_benchmark_accuracy, label="Benchmark accuracy")
            plt.plot(index_variable, avg_rl_accuracy, label="RL accuracy")

        plt.axhline(y=np.median(avg_benchmark_accuracy), c="cornflowerblue")
        plt.axhline(y=np.median(avg_rl_accuracy), c="orange")
        plt.xlabel("Number of variables")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.gca().invert_xaxis()
        plt.show()

        return is_better_list

    def get_best_state(self) -> tuple[list[State | float], list[State | float]]:
        best_v_value = float("-inf")
        best_state_v_value = None

        best_reward = float("-inf")
        best_state_reward = None

        for depth, states in self.feature_structure.items():
            if depth == 0:
                continue
            for state in states:
                reward = float(state.reward) if state.reward is not None else float("-inf")
                if reward > best_reward:
                    best_reward = reward
                    best_state_reward = state
                if state.v_value > best_v_value:
                    best_v_value = state.v_value
                    best_state_v_value = state

        return [best_state_reward, best_reward], [best_state_v_value, best_v_value]
