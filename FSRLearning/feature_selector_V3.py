from dataclasses import dataclass
from operator import index

#Maths and others
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import itertools

#Other class
from feature_selection_RL_V3 import *

#ML Sklearn library
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@dataclass
class Feature_Selector_RL:
    '''
        This is the class used to create a feature selector with the RL method

        fit enable to get the results structured as follows:
            [
                Feature index : list,
                Number of times a feature has been played: list
                AOR value per feature: list,
                Sorted list of feature from the less to the most important for the model: list
            ]

        Parameters explaination:
            [
                Alpha [0; 1] : rate of updates
                Gamma [0; 1] : discount factor to moderate the effect of observing the next state (0=shortsighted; 1=farsighted)
                Starting_state : string empty of random --> if empty then the starting state is empty elif random we start from a random state
            ]
    '''

    feature_number: int
    nb_explored: list = None
    nb_not_explored: list = None
    feature_structure: dict = None
    aor: list = None
    eps: float = .1
    alpha: float = .5
    gamma: float = .70
    nb_iter: int = 100
    explored: int = 0
    not_explored: int = 0
    starting_state: str = 'empty'


    def fit_predict(self, X, y, clf = RandomForestClassifier(max_depth=4)) -> tuple([list, float]):
        '''
            Get the sorted weighted variables

            Input : 
            [
                X, y : input data
                clf : classifier used for reward evaluation
            ]
        '''

        #We init the process
        print('---------- Default Parameters init ----------')
        self.aor = [np.zeros(self.feature_number), np.zeros(self.feature_number)]
        self.feature_structure = {}
        self.nb_explored = []
        self.nb_not_explored = []

        print('---------- Process init ----------')
        feature_selection_process = FeatureSelectionProcessV3(self.feature_number, self.eps, self.alpha, self.gamma, self.aor, self.feature_structure)

        print('---------- Data Processing ----------')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        print('---------- The process has been successfully init ----------')

        print('---------- Training ----------')

        #We iterate it times on the graph to get informations about each feature
        for it in tqdm(range(self.nb_iter)):
            #We start from the empty state
            if self.starting_state == 'random':
                init = feature_selection_process.pick_random_state()
            elif self.starting_state == 'empty':
                init = feature_selection_process.start_from_empty_set()

            #current_state = init[0]
            current_state, is_empty_state = init[0], init[1]

            #Informations about the current state
            current_state_depth: int = current_state.number[0]

            #We init the worsen state stop condition
            not_stop_worsen: bool = True
            previous_v_value, nb_worsen = current_state.v_value, 0
        
            while not_stop_worsen:
                #Worsen condition update
                previous_v_value = current_state.v_value

                #We select the next state with a stop condition if we reach the final state
                if current_state.number[0] == self.feature_number:
                    not_stop_worsen = False
                else:
                    next_step = current_state.select_action(feature_selection_process.feature_structure, feature_selection_process.eps, feature_selection_process.aor, is_empty_state)
                    next_action, next_state, was_eps = next_step[0], next_step[1], next_step[2]

                #We count the explored state
                if was_eps:
                    self.explored += 1
                else:
                    self.not_explored += 1

                #We evaluate the reward of the next_state
                next_state.get_reward(X_train, y_train, X_test, y_test, clf)

                #We update the v_value of the current_state
                current_state.update_v_value(.99, .99, next_state)

                #We update the worsen stop condition
                #We set the variables for the worsen v_value state stop condition
                if previous_v_value > current_state.v_value and not is_empty_state:
                    if nb_worsen < round(np.sqrt(self.feature_number)):
                        nb_worsen += 1
                    else:
                        not_stop_worsen = False

                #We update the aor table
                feature_selection_process.aor = next_action.get_aorf(feature_selection_process.aor)

                #print(f'current state after {current_state}')

                #We add these information to the history (the graph)
                feature_selection_process.add_to_historic(current_state)

                #We change the current_state with the new one
                current_state = next_state
                
                #We update the current_state's depth
                current_state_depth = current_state.number[0]

                if current_state_depth >= self.feature_number:
                    not_stop_worsen = False

                is_empty_state = False
            #print(f'{feature_selection_process.feature_structure}')
                
            self.nb_explored.append(self.explored)
            self.nb_not_explored.append(self.not_explored)

        results = feature_selection_process.get_final_aor_sorted()

        print('---------- Results ----------')
        return results, self.nb_explored[-1] + self.nb_not_explored[-1]

    def get_plot_ratio_exploration(self):
        '''
            Plots the graph of the evolution of the already and newly visited states
        '''
        plt.plot([i for i in range(len(self.nb_not_explored))], self.nb_not_explored, label='Already explored State')
        plt.plot([i for i in range(len(self.nb_explored))], self.nb_explored, label='Explored State')
        plt.xlabel('Number of iterations')
        plt.ylabel('Number of states')
        plt.legend(loc="upper left")

        plt.show()

    def get_feature_strengh(self, results):
        '''
            Plots the graph of the relative strengh of each variable
        '''
        #Relative strengh of the variable
        plt.bar(x = results[0][0], height=results[0][2], color=['blue' if rew >=0 else 'red' for rew in results[0][2]])
        plt.xlabel('Feature in the dataset')
        plt.ylabel('Average of the reward brought by the feature')

        plt.show()

    def get_depth_of_visited_states(self):
        '''
            Plot the evolution of the size of the visited states in function of the iterations
        '''
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


    def compare_with_benchmark(self, X, y, results) -> list:
        '''
            Returns all the metrics at each iteration on the set of feature

            Plots the graph of these evolutions
        '''
        is_better_list: list = []
        avg_benchmark_acccuracy: list = []
        avg_rl_acccuracy: list = []

        results = results[0]

        print('---------- Data Processing ----------')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        print('---------- Score ----------')
        for i in range(1, self.feature_number):
            #From RL
            clf = RandomForestClassifier(max_depth=4)
            clf.fit(X_train[results[-1][i:]], y_train)
            accuracy: float = clf.score(X_test[results[-1][i:]], y_test)

            #Benchmark
            estimator = RandomForestClassifier(max_depth=4)
            selector = RFE(estimator, n_features_to_select=len(results[-1]) - i, step=1)
            selector = selector.fit(X_train, y_train)
            sele_acc = selector.score(X_test, y_test)

            if accuracy >= sele_acc:
                is_better_list.append(1)
            else:
                is_better_list.append(0)
            
            avg_benchmark_acccuracy.append(sele_acc)
            avg_rl_acccuracy.append(accuracy)

            print(f'Set of variables : Benchmark : {X_train.columns[selector.support_].tolist()} and RL : {results[-1][i:]}')
            print(f'Benchmark accuracy : {sele_acc}, RL accuracy : {accuracy} with {len(results[-1]) - i} variables {is_better_list}')
        
        print(f'Average benchmark accuracy : {np.mean(avg_benchmark_acccuracy)}, rl accuracy : {np.mean(avg_rl_acccuracy)}')
        print(f'Median benchmark accuracy : {np.median(avg_benchmark_acccuracy)}, rl accuracy : {np.median(avg_rl_acccuracy)}')
        print(f'Probability to get a set of variable with a better metric than RFE : {np.sum(is_better_list) / len(is_better_list)}')
        print(f'Aread between the two curves : {np.trapz(avg_rl_acccuracy) - np.trapz(avg_benchmark_acccuracy)}')

        index_variable: list = [i for i in range(len(avg_benchmark_acccuracy))]

        avg_benchmark_acccuracy.reverse()
        avg_rl_acccuracy.reverse()

        #Smooth the curve for a better visual aspect
        avg_benchmark_acccuracy_smooth = make_interp_spline(index_variable, avg_benchmark_acccuracy)
        avg_rl_acccuracy_smooth = make_interp_spline(index_variable, avg_rl_acccuracy)

        X_benchmark = np.linspace(np.min(index_variable), np.max(index_variable), 100)
        Y_benchmark= avg_benchmark_acccuracy_smooth(X_benchmark)
        Y_RL= avg_rl_acccuracy_smooth(X_benchmark)

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

    def get_best_state(self) -> tuple([[float, State], [float, State]]):
        '''
            Returns the optimal state

            Returns : Tuple(Best_rewarded_state, Best_feature_set)
        '''

        best_v_value: float = 0
        best_state_v_value: State = None

        best_reward: float = 0
        best_state_reward: State = None

        #Dictionary browsing by key
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
