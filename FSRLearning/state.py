from action import Action

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier, is_regressor

class State:
    '''
        State object

        number: position in the dictionary of the graph
        description: represents the set of feature in the set
        v_value: V value of the state
        nb_visited: number of times that the set has been visited
    '''
    def __init__(self, 
                 number: list,
                 description: list,
                 v_value: float,
                 reward: float = 0,
                 nb_visited: int = 0) -> None:
        self.number = number
        self.description = description
        self.v_value = v_value
        self.reward = reward
        self.nb_visited = nb_visited

    def get_reward(self, clf, X, y) -> float:
        '''
            Returns the reward of a set of variable

            clf: type of the classifier with which we want to evaluate the data
        '''
        # Train classifier with state_t variable and state t+1 variables and compute the diff of the accuracy
        if self.reward == 0:
            if self.description == []:
                self.reward = 0
                return 0
            else:
                # The state has never been visited and init the reward
                df = pd.concat([X.iloc[:, self.description], y], axis = 1)
                df = df.drop_duplicates(ignore_index = True) 

                if is_classifier(clf):
                    min_samples = np.min(np.array(df.iloc[:, -1].value_counts()))
                    if min_samples < 5 and min_samples >= 2:
                        accuracy: float = np.mean(cross_val_score(clf, df.iloc[:, :-1], df.iloc[:, -1], cv = min_samples, scoring = 'balanced_accuracy'))
                    elif min_samples < 2:
                        accuracy: float = 0
                    else:
                        accuracy: float = np.mean(cross_val_score(clf, df.iloc[:, :-1], df.iloc[:, -1], cv = 5, scoring = 'balanced_accuracy'))
                elif is_regressor(clf):
                    accuracy: float = np.mean(cross_val_score(clf, df.iloc[:, :-1], df.iloc[:, -1], cv = 5, scoring = 'r2'))

                self.reward = accuracy
                return self.reward
        else:
            return self.reward

    def select_action(self, feature_structure: dict, eps: float, aorf_histo: list, is_empty_state: bool):
        ''' 
            Returns an action object

            feature_structure: current dictionnary of the structure of the graph
            eps: probability of choosing a random action [between 0 and 1]

            This method enables to train only once a model and get the accuracy
        '''
        #We get the neighboors
        get_neigh: list = self.get_neighboors(feature_structure, [i for i in range(0, len(aorf_histo[0]))])

        #Check if it is an eps step
        eps_step: bool = bool(np.random.binomial(1, eps))

        if self.nb_visited == 0 or eps_step:
            has_explored_node: list = [neigh for neigh in get_neigh if neigh.v_value != 0]
            if not has_explored_node or eps_step:
                #We select not explored state randomly
                next_state: State = np.random.choice(get_neigh)

                return Action(self, next_state), next_state, True
            else:
                #Get argmax next state
                next_state = self.get_argmax(get_neigh, aorf_histo)

                return Action(self, next_state), next_state, False
        else:
            #Get argmax next state
            next_state = self.get_argmax(get_neigh, aorf_histo)

            return Action(self, next_state), next_state, False
        
    def get_argmax(self, get_neigh: list, aorf_histo):
        '''
            Returns the argmax of the list of neighbors 

            get_neigh: list of the neighbors of the self state
            aorf_histo: value of the aor
        '''
        #We select a state where the possible next feature has the maximum AORf
        possible_feature: list = [list(set(neigh.description) - set(self.description))[0] for neigh in get_neigh]
    
        #We determine the max AORf
        feature_max_aorf: list = np.argmax([aorf_histo[1][feat] for feat in possible_feature])

        #We get the max feature
        next_feature: int = possible_feature[feature_max_aorf]

        #We get the next state
        next_state: State = [
            neigh for neigh in get_neigh
            if list(set(neigh.description)-set(self.description)) == [next_feature]
        ][0]

        return next_state

    def get_neighboors(self, feature_structure: dict, feature_list: list) -> list:
        '''
            Returns the list of the neighboors of the current state

            feature_structure: current dictionnary of the structure of the graph
            feature_list: list of the int identifiers of the features in the data set (len = number of features in the datas set)
        '''
        neigh_depth_graph: int = self.number[0] + 1

        if neigh_depth_graph in feature_structure: 
            existing_neigh: list = [
                state_neigh for state_neigh in feature_structure[neigh_depth_graph]
                if len(list(set(state_neigh.description)-set(self.description))) == 1
            ]

            possible_neigh: list = [
                State([neigh_depth_graph, len(feature_structure[neigh_depth_graph])], self.description + [feature], 0)
                for feature in feature_list
                if feature not in self.description and not np.any([set(self.description + [feature])==set(neigh.description) for neigh in existing_neigh])
            ]

            return existing_neigh + possible_neigh
        else:
            possible_neigh: list = [
                State([neigh_depth_graph, 0], self.description + [feature], 0)
                for feature in feature_list
                if feature not in self.description
            ]
        
            return possible_neigh

    def update_v_value(self, alpha: float, gamma: float, next_state) -> float:
        '''
            Update the v_value of a state

            Alpha [0; 1] : rate of updates
            Gamma [0; 1] : discount factor to moderate the effect of observing the next state (0=shortsighted; 1=farsighted)
            next_state: the next state that has been chosen by the eps_greedy algorithm

            Returns a float number
        '''
        self.v_value += alpha * ((next_state.reward - self.reward) + gamma * next_state.v_value - self.v_value)   
        
    def is_final(self, nb_of_features: int) -> bool:
        '''
            Check if a state is a final state (with all the features in the state)

            nb_of_features: number of features in the data set 

            Returns True if all the possible features are in the state
        '''
        if len(self.description) == nb_of_features:
            return True
        else:
            return False

    def is_equal(self, compared_state) -> bool:
        '''
            Compare if two State objects are equal

            compared_state: state to be compared with the self state

            Returns True if yes else returns False
        '''
        if set(self.description) == set(compared_state.description):
            return True
        else:
            return False
