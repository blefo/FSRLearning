from dataclasses import dataclass
import itertools
from os import stat
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@dataclass
class State:
    '''
        State object
    '''
    number: list
    description: list
    v_value: float
    reward: float = 0
    nb_visited: int = 0

    def get_reward(self, X_train, y_train, X_test, y_test) -> float:
        '''
            Return the reward of a set of variable
        '''
        #Train classifier with state_t variable and state t+1 variables and compute the diff of the accuracy
        if self.reward == 0:
            clf = RandomForestClassifier(max_depth=2)
            clf.fit(X_train, y_train)
            accuracy: float = clf.score(X_test, y_test)
            self.reward = accuracy
            return accuracy
        else:
            return self.reward

    def select_action(self, feature_structure: dict, eps: float, aorf_histo: list):
        '''
            Return an action object

            This method enables to train only once a model and get the accuracy
        '''
        #Get possible neighboors of the current state
        state_neigh = self.get_neighboors(feature_structure, [i for i in range(len(aorf_histo[0]))])

        #Random state selection
        select_random_element: int = np.random.randint(len(state_neigh))
        next_state: State = None if not state_neigh else state_neigh[select_random_element]

        #Get current state object
        current_state: State = self

        #We init the discovering rate
        e_greedy_choice: bool = bool(np.random.binomial(1, eps))

        if self.v_value == 0 or next_state is None or e_greedy_choice:
            #State never visited ==> we chose a random action
            select_random_element: int = np.random.randint(len(state_neigh))

            next_state: State = state_neigh[select_random_element]

            #We update the number of visit
            self.nb_visited += 1

            return Action(current_state, next_state), next_state
        else:
            #Get AOR of a variable and select the max AOR associated to a variable
            if self.number[0] <= 11:
                get_existing_neigh: list = [neigh for neigh in feature_structure[self.number[0] + 1] if np.all([j in neigh.description for j in self.description])]

                get_aorf_neigh: list = [list(set(feat_added.description)-set(self.description)) for feat_added in get_existing_neigh]
                get_aorf_max: list = np.argmax([aorf_histo[1][feat] for feat in get_aorf_neigh])

                next_state: State = [st_max for st_max in get_existing_neigh if list(set(st_max.description)-set(self.description)) == get_aorf_neigh[get_aorf_max]][0]
            else:
                next_state: State = State([13, 0], [i for i in range(len(aorf_histo[0]))], 0)
            #We update the number of visit
            self.nb_visited += 1

            return Action(current_state, next_state), next_state
            

    def update_v_value(self, alpha: float, gamma: float, v_value_next_state: float):
        '''
            Update the v_value of a state
        '''
        self.v_value += alpha * (self.reward + gamma * v_value_next_state - self.v_value)

    def get_neighboors(self, feature_structure: dict, feature_list: int) -> list:
        '''
            Returns the list of the neighboors of the current state
        '''
        graph_depth: int = self.number[0]

        #Check if the graph has a layer at this depth
        if graph_depth in feature_structure:
            #There is a graph_depth n° layer
            all_possible_states: list = [
                State(
                    [graph_depth+1, len(feature_structure[graph_depth])],
                    list(combin),
                    0
                )
            for combin in itertools.combinations(feature_list, graph_depth + 1) if np.all([j in combin for j in self.description])]
            
            if graph_depth <= 11:
                existing_states: list = [neigh for neigh in feature_structure[graph_depth + 1] if np.all([j in neigh.description for j in self.description])]
            else:
                existing_states: list = []

            final_neigh: list = all_possible_states + existing_states
        else:
            final_neigh: list = [
                State(
                    [graph_depth+1, 0],
                    list(combin),
                    0
                )
            for combin in itertools.combinations(feature_list, graph_depth + 1) if np.all([j in combin for j in self.description])]
        
        return final_neigh

            

    def is_final(self, nb_of_features: int) -> bool:
        '''
            Check if a state is a final state (with all the features in the state)

            Returns True if all the possible features are in the state
        '''
        if len(self.description) == nb_of_features:
            return True
        else:
            return False

    def is_equal(self, compared_state) -> bool:
        '''
            Compare if two State objects are equal

            Returns True if yes else returns False
        '''
        if np.array_equal(self.description, compared_state.description) and np.array_equal(self.number, compared_state.number):
            return True
        else:
            return False

@dataclass
class Action:
    '''
        Action Object
    '''
    state_t: State
    state_next: State              

    def get_aorf(self, aor_historic: list) -> float:
        '''
        Update the ARO of a feature

        Return the AOR table
        '''

        #Get historic
        chosen_feature: int = list(set(self.state_next.description) - set(self.state_t.description))
        aorf_nb_played_old: int = aor_historic[0][int(chosen_feature[0])]
        aorf_value_old: int = aor_historic[1][int(chosen_feature[0])]

        #Update the aor for the selection of f
        aor_historic[0][int(chosen_feature[0])] += 1  

        #Update the aor for the selection of f
        aor_historic[1][int(chosen_feature[0])] = ((aorf_nb_played_old - 1) * aorf_value_old + self.state_t.v_value) / (aorf_nb_played_old + 1)

        return aor_historic          

@dataclass
class FeatureSelectionProcessV2:
    '''
        Init aor list such that aor = [[np.zeros(nb_of_features)], [np.zeros(nb_of_features)]]

    '''
    nb_of_features: int
    eps: float
    alpha: float
    gamma: float

    #Memory of AORf [[nb f is selected], [AOR value]]
    aor: list

    #Init the state structure
    feature_structure: dict


    def pick_random_state(self) -> State:
        '''
            Select a random state in all the possible state space
            
            Return a state randomly picked
        '''
        random_start: int = np.random.randint(1, self.nb_of_features)
        random_end: int = np.random.randint(random_start, self.nb_of_features)
        chosen_state: list = np.random.default_rng(seed=42).permutation([var for var in range(self.nb_of_features)])[random_start:random_end]

        depth: int = len(chosen_state)

        #Check if the dict is empty
        if self.feature_structure.get(depth) is not None:
            return State([depth, len(self.feature_structure.get(depth))], chosen_state, 0, 0)
        else:
            return State([depth, 0], chosen_state, 0, 0)

    def start_from_empty_set(self) -> State:
        '''
            Start from the empty set (with no feature selected)
            
            Returns the empty initial state
        '''
        depth = 0
        if not bool(self.feature_structure):
            return State([0, 0], [], 0, 0)
        else:
            return self.feature_structure[depth][0]

    def add_to_historic(self, visited_state: State):
        '''
            Add to the feature structure historic function
        '''
        depth = visited_state.number[0]
        if depth in self.feature_structure:
            is_in_table: bool = [visited_state.is_equal(state) for state in self.feature_structure[depth]]
            get_index_where_true: int = [is_in_table.index(i) for i in is_in_table if i] if np.any(is_in_table) else None
            if get_index_where_true is not None:
                self.feature_structure[depth][get_index_where_true[0]] = visited_state
            else:
                self.feature_structure[depth].append(visited_state)
        else:
            self.feature_structure[depth] = [visited_state]

    def get_final_aor_sorted(self) -> list:
        '''
            Returns the aor table sorted by ascending

            Index of the feature
            Number of time the feature has been played
            Value of the feature
            Best feature (from the lowest to the biggest)
        '''

        index: list = [i for i in range(self.nb_of_features)]
        nb_played: list = self.aor[0]
        values: list = self.aor[1]

        index, values = zip(*sorted(zip(index, values)))

        return [index, nb_played, values, np.argsort(self.aor[1])]

    def get_optimal_state_value(self) -> State:
        '''
            Returns the optimal state
        '''

        optimal_state: State = State([0, 0], [], 0)

        #Dictionary browsing by key
        for key in self.feature_structure.keys():
            if key == 0:
                pass
            else:
                for value in self.feature_structure[key]:
                    if value.v_value > optimal_state.v_value:
                        optimal_state = value

        return optimal_state
    


    
    

    

    