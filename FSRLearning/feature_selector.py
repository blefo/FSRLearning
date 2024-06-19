from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from .state import State
from .fsrlearning import FeatureSelectionProcess

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.base import is_classifier, is_regressor

class FeatureSelectorRL:
    """Class for feature selector using the RL method."""
        
    def __init__(self, feature_number: int, eps: float = .1, alpha: float = .5, gamma: float = .70, nb_iter: int = 100, 
                 starting_state: str = 'empty', nb_explored: list = None, nb_not_explored: list = None, 
                 feature_structure: dict = None, aor: list = None, explored: int = 0, not_explored: int = 0):
        """
        Constructor for FeatureSelectorRL.

        Parameters
        ----------
        feature_number : integer
            Number of features.
        eps : float [0, 1], default = 0.1
            Probability of choosing a random next state. 0 is an only greedy algorithm and 1 is an only random algorithm.
        alpha : float [0, 1], default = 0.5
            Controls the rate of updates. 0 is a very not updating state and 1 is a very updating state.
        gamma : float [0, 1], default = 0.7
            Discount factor to moderate the effect of observing the next state. 0 exhibits shortsighted behavior and 1 exhibits farsighted behavior.
        nb_iter : integer, default = 100
            Number of sequences to go through the graph.
        starting_state : {"empty", "random"}, default = "empty"
            Starting state of the algorithm. 

            If "empty", the algorithm starts from an empty state.
            If "random", the algorithm starts from a random state in the graph.
        """
                     
        self.feature_number = feature_number
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.nb_iter = nb_iter
        self.starting_state = starting_state
        self.nb_explored = nb_explored
        self.nb_not_explored = nb_not_explored
        self.feature_structure = feature_structure
        self.aor = aor
        self.explored = explored
        self.not_explored = not_explored

    def fit_predict(self, X, y, estimator):
        """        
        Fit the FeatureSelectorRL algorithm according to the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data vector, where `n_samples` is the number of samples and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target values.
        estimator : Classifier or Regressor Estimator 
            A supervised learning estimator with a ``fit`` method. Used for reward evaluation.

        Returns
        ----------
        results : tuple
            Output of the selection process (2-object tuple):
                List:
                    Index of the features that have been sorted.
                    Number of times that each feature has been chosen.
                    Mean reward brought by each feature.
                    Ranking of the features from the less important to the most important.
                Integer:
                    Number of states visited.
        """

        # Init the process
        print('---------- Default Parameters init ----------')
        self.aor = [np.zeros(self.feature_number), np.zeros(self.feature_number)]
        self.feature_structure = {}
        self.nb_explored = []
        self.nb_not_explored = []

        print('---------- Process init ----------')
        feature_selection_process = FeatureSelectionProcess(self.feature_number, self.eps, self.alpha, self.gamma,
                                                            self.aor, self.feature_structure)

        print('---------- Data Processing ----------')
        X = pd.DataFrame(X)
        y = pd.Series(y)

        print('---------- The process has been successfully init ----------')

        print('---------- Training ----------')

        # We iterate it times on the graph to get informations about each feature
        for it in tqdm(range(self.nb_iter)):
            # Start from the empty state
            if self.starting_state == 'random':
                init = feature_selection_process.pick_random_state()
            elif self.starting_state == 'empty':
                init = feature_selection_process.start_from_empty_set()

            # current_state = init[0]
            current_state, is_empty_state = init[0], init[1]

            # Information about the current state
            current_state_depth: int = current_state.number[0]

            # Init the worsen state stop condition
            not_stop_worsen: bool = True
            previous_v_value, nb_worsen = current_state.v_value, 0

            while not_stop_worsen:
                # Worsen condition update
                previous_v_value = current_state.v_value

                # We select the next state with a stop condition if we reach the final state
                if current_state.number[0] == self.feature_number:
                    not_stop_worsen = False
                else:
                    next_step = current_state.select_action(feature_selection_process.feature_structure,
                                                            feature_selection_process.eps,
                                                            feature_selection_process.aor, is_empty_state)
                    next_action, next_state, was_eps = next_step[0], next_step[1], next_step[2]

                # We count the explored state
                if was_eps:
                    self.explored += 1
                else:
                    self.not_explored += 1

                # We evaluate the reward of the next_state
                next_state.get_reward(estimator, X, y)

                # We update the v_value of the current_state
                current_state.update_v_value(.99, .99, next_state)

                # We update the worsen stop condition
                # We set the variables for the worsen v_value state stop condition
                if previous_v_value > current_state.v_value and not is_empty_state:
                    if nb_worsen < round(np.sqrt(self.feature_number)):
                        nb_worsen += 1
                    else:
                        not_stop_worsen = False

                # We update the aor table
                feature_selection_process.aor = next_action.get_aorf(feature_selection_process.aor)

                # We add these information to the history (the graph)
                feature_selection_process.add_to_historic(current_state)

                # We change the current_state with the new one
                current_state = next_state

                # We update the current_state's depth
                current_state_depth = current_state.number[0]

                if current_state_depth >= self.feature_number:
                    not_stop_worsen = False

                is_empty_state = False
            # print(f'{feature_selection_process.feature_structure}')

            self.nb_explored.append(self.explored)
            self.nb_not_explored.append(self.not_explored)

        results = feature_selection_process.get_final_aor_sorted()

        print('---------- Results ----------')
        return results, self.nb_explored[-1] + self.nb_not_explored[-1]

    def get_plot_ratio_exploration(self):
        """
        Plot a graph comparing the number of already visited nodes and visited nodes.
        """
        
        plt.plot([i for i in range(len(self.nb_not_explored))], self.nb_not_explored, label='Already explored State')
        plt.plot([i for i in range(len(self.nb_explored))], self.nb_explored, label='Explored State')
        plt.xlabel('Number of iterations')
        plt.ylabel('Number of states')
        plt.legend(loc="upper left")

        plt.show()

    def get_feature_strength(self, results):
        """
        Plot a graph of the relative impact of each feature on the model.

        Parameter
        ----------
        results : 2-object tuple
            Results returned from fit_predict.
        """
        
        #Relative strengh of the variable
        plt.bar(x=results[0][0], height=results[0][2], color=['blue' if rew >= 0 else 'red' for rew in results[0][2]])
        plt.xlabel('Feature\'s name')
        plt.ylabel('Average feature reward')

        plt.show()

    def get_depth_of_visited_states(self):
        """
        Plot a graph of the number of times that a state of a certain size has been visited.
        """
        
        sum_depth = []
        for key in self.feature_structure:
            #Browse every state with one size in the graph
            sum_index = []
            for st in self.feature_structure[key]:
                sum_index.append(st.nb_visited)

            sum_depth.append(np.sum(sum_index))

        plt.plot([i for i in range(len(sum_depth))], sum_depth)
        plt.xlabel('Size of the visited states')
        plt.ylabel('Number of visits')
        plt.plot()

    def compare_with_benchmark(self, X, y, results, estimator):
        """
        Compare the performance of FeatureSelectorRL with RFE from Sickit-Learn.
        Return balanced accuracy score for classifiers and r2 score for regressors at each iteration 
        and plot the graph of these evolutions.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data vector, where `n_samples` is the number of samples and `n_features` is the number of features.
            Same as the X passed to fit_predict.
        y : array-like of shape (n_samples,)
            Target values. Same as the y passed to fit_predict.
        results : 2-object tuple
            Results returned from fit_predict.
        estimator : Classifier or Regressor Estimator
            A supervised learning estimator with a ``fit`` method that provides information about feature importance 
            (e.g. `coef_`, `feature_importances_`).
        """
        
        is_better_list: list = []
        avg_benchmark_acccuracy: list = []
        avg_rl_acccuracy: list = []

        results = results[0]

        print('---------- Data Processing ----------')
        X = pd.DataFrame(X)
        y = pd.Series(y)

        print('---------- Score ----------')
        for i in range(1, self.feature_number):
            # From RL
            if is_classifier(estimator):
                min_samples = y.value_counts().min()
                if min_samples >= 5:
                    accuracy = np.mean(cross_val_score(estimator, X.iloc[:, results[-1][i:]], y, cv = 5, scoring = 'balanced_accuracy'))
                elif min_samples < 5 and min_samples >= 2:
                    accuracy = np.mean(cross_val_score(estimator, X.iloc[:, results[-1][i:]], y, cv = min_samples, scoring = 'balanced_accuracy'))
                else:
                    accuracy = 0
            elif is_regressor(estimator):
                num_samples = len(y)
                if num_samples >= 10:
                    accuracy = np.mean(cross_val_score(estimator, X.iloc[:, results[-1][i:]], y, cv = 5, scoring = 'r2'))
                elif num_samples < 10 and num_samples >= 4:
                    accuracy = np.mean(cross_val_score(estimator, X.iloc[:, results[-1][i:]], y, cv = num_samples // 2, scoring = 'r2'))
                else:
                    accuracy = 0
            else:
                raise TypeError("The provided estimator is neither a classifier nor a regressor. Please make sure to pass a classifier or regressor to the method.")
            
            if np.isnan(accuracy):
                accuracy = 0
            
            # Benchmark
            selector = RFE(estimator, n_features_to_select=len(results[-1]) - i, step=1)
            if is_classifier(estimator):
                cv_results = cross_validate(selector, X, y, cv=5, scoring='balanced_accuracy', return_estimator=True)
            elif is_regressor(estimator):
                cv_results = cross_validate(selector, X, y, cv=5, scoring='r2', return_estimator=True)
            else:
                raise TypeError("The provided estimator is neither a classifier nor a regressor. Please make sure to pass a classifier or regressor to the method.")
            
            sele_acc = np.mean(cv_results['test_score'])
            if np.isnan(sele_acc):
                sele_acc = 0

            if accuracy >= sele_acc:
                is_better_list.append(1)
            else:
                is_better_list.append(0)

            avg_benchmark_acccuracy.append(sele_acc)
            avg_rl_acccuracy.append(accuracy)

            print(
                f"Set of variables : Benchmark (For Each Fold): {[X.columns[selector.support_].tolist() for selector in cv_results['estimator']]} and RL : {results[-1][i:]}")
            print(
                f'Benchmark accuracy : {sele_acc}, RL accuracy : {accuracy} with {len(results[-1]) - i} variables {is_better_list}')

        print(
            f'Average benchmark accuracy : {np.mean(avg_benchmark_acccuracy)}, rl accuracy : {np.mean(avg_rl_acccuracy)}')
        print(
            f'Median benchmark accuracy : {np.median(avg_benchmark_acccuracy)}, rl accuracy : {np.median(avg_rl_acccuracy)}')
        print(
            f'Probability to get a set of variable with a better metric than RFE : {np.sum(is_better_list) / len(is_better_list)}')
        print(f'Aread between the two curves : {np.trapz(avg_rl_acccuracy) - np.trapz(avg_benchmark_acccuracy)}')

        index_variable: list = [i for i in range(len(avg_benchmark_acccuracy))]

        avg_benchmark_acccuracy.reverse()
        avg_rl_acccuracy.reverse()

        # Smooth the curve for a better visual aspect
        avg_benchmark_acccuracy_smooth = make_interp_spline(index_variable, avg_benchmark_acccuracy)
        avg_rl_acccuracy_smooth = make_interp_spline(index_variable, avg_rl_acccuracy)

        X_benchmark = np.linspace(np.min(index_variable), np.max(index_variable), 100)
        Y_benchmark = avg_benchmark_acccuracy_smooth(X_benchmark)
        Y_RL = avg_rl_acccuracy_smooth(X_benchmark)

        plt.axhline(y=np.median(avg_benchmark_acccuracy), c='cornflowerblue')
        plt.axhline(y=np.median(avg_rl_acccuracy), c='orange')
        plt.plot(X_benchmark, Y_benchmark, label='Benchmark acccuracy')
        plt.plot(X_benchmark, Y_RL, label='RL accuracy')
        plt.xlabel('Number of variables')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right")
        plt.gca().invert_xaxis()

        plt.show()

        return is_better_list

    def get_best_state(self):
        """
        Return the optimal state.
        
        Returns
        ----------
        state : tuple
            2-object tuple:
                List:
                    Best state reward.
                    Best reward. 
                List:
                    Best state v value.
                    Best v value.
        """

        best_v_value: float = 0
        best_state_v_value: State = None

        best_reward: float = 0
        best_state_reward: State = None

        # Dictionary browsing by key
        for key in self.feature_structure:
            if key == 0:
                pass
            else:
                for value in self.feature_structure[key]:
                    if value.reward > best_reward:
                        best_reward = value.reward
                        best_state_reward = value
                    if value.v_value > best_v_value:
                        best_v_value = value.v_value
                        best_state_v_value = value

        return [best_state_reward, best_reward], [best_state_v_value, best_v_value]
