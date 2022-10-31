from dataclasses import dataclass
import numpy as np
import itertools

def feature_sub_set_generator(feature_list: list) -> dict:
    final_dict: dict = {}
    len_feature_list: int = len(feature_list)

    for set_size in range(len_feature_list):
        final_dict[set_size] = []
        for permu in itertools.permutations(feature_list, set_size):
            final_dict[set_size].append(list(permu))

    final_dict[len_feature_list] = feature_list
    
    return final_dict

@dataclass
class State:
    number: list
    v_value: float

    def select_action(self, state_structure: dict, all_states: list, eps: float, aorf_histo: list) -> list:
        #Get possible neighbors of the current state
        state_neigh: list = [pos for pos, neigh in enumerate(state_structure[self.number[0] + 1]) if neigh[-2] == state_structure[self.number[0]][self.number[1]][0]]

        #Randomly selects an index where the next feature is selected
        random_state_selector: int = np.random.choice(state_neigh)
        number_state_next: list = [state_structure[self.number[0] + 1], random_state_selector]

        #Convert position to state object
        current_state: State = [selected_state for selected_state in all_states if selected_state.number == self.number][0]
        next_state: State = [selected_state for selected_state in all_states if selected_state.number == number_state_next][0]

        if self.v_value == 0:
            return Action(current_state, next_state)

        else:
            e_greedy_choice: bool = bool(np.random.binomial(1, eps))
            if e_greedy_choice:
                return Action(current_state, next_state)
            else:
                possible_actions: list = [Action(current_state, [selected_state for selected_state in all_states if selected_state.number == neigh][0]) for neigh in state_neigh]
                reward_hist: list = [Action(current_state, [selected_state for selected_state in all_states if selected_state.number == neigh][0]).get_aorf(aorf_histo, state_structure) for neigh in state_neigh]
                better_state: State = np.argmax(reward_hist)

                return Action(current_state, possible_actions[better_state])

    def update_v_value(self, alpha: float, gamma: float, reward_action: float, v_value_next_state: float):
        self.v_value += alpha * (reward_action + gamma * v_value_next_state - self.v_value)

@dataclass
class Action:
    state_t: State
    state_next: State
    reward: float = 0

    def get_reward(self, classifier) -> float:
        #Train classifier with state_t variable and state t+1 variables and compute the diff of the accuracy
        raise NotImplementedError               

    def get_aorf(self, aor_historic: list, state_structure: list) -> float:
        #Get state position in state_structure
        depth: int = self.state_t.number[0]
        height: int = self.state_t.number[1]
        feature_chosen: int = state_structure[depth][height][-1]

        #Get historic
        aorf_nb_played_old: int = aor_historic[0][feature_chosen]
        aorf_value_old: int = aor_historic[0][feature_chosen]

        #Update the aor for the selection of f
        aor_historic[0][feature_chosen] += 1  

        #Update the aor for the selection of f
        aor_historic[1][feature_chosen] = ((aorf_nb_played_old - 1) * aorf_value_old + self.state_t.v_value) / aorf_nb_played_old

        return aor_historic[1][feature_chosen]            

@dataclass
class FeatureSelectionProcess:
    '''
        Init aor list such that aor = [[np.zeros(nb_of_features)], [np.zeros(nb_of_features)]]
        Init feature_structure dict such that feature_structure = feature_sub_set_generator([feature_name for feature_name in range(nb_of_features)])
    '''
    nb_of_features: int
    eps: float
    alpha: float
    gamma: float

    #Memory of AORf [[nb f is selected], [AOR value]]
    aor: list

    #Init the state structure
    feature_structure: dict

    def init_states(self) -> list:
        states_lists: list = []
        for nb in range(self.nb_of_features):
            for nb_under in range(self.feature_structure[nb]):
                states_lists.append(State([nb, nb_under], 0))

        return states_lists

    
    

    

    