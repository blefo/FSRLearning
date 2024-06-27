from .state import State
import numpy as np

class FeatureSelectionProcess:
    """
    Init aor list such that aor = [[np.zeros(nb_of_features)], [np.zeros(nb_of_features)]].

    nb_of_features: Number of feature in the data set
    eps: probability of choosing a random action (uniform or softmax)
    alpha: Controls the rate of updates. 0 is a very not updating state and 1 is a very updating state
    gamma: Discount factor to moderate the effect of observing the next state. 0 exhibits shortsighted behavior and 1 exhibits farsighted behavior
    """

    def __init__(self,
                 nb_of_features: int,
                 eps: float,
                 alpha: float,
                 gamma: float,
                 aor: list,
                 feature_structure: dict) -> None:
        self.nb_of_features = nb_of_features
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

        #Memory of AORf [[nb f is selected], [AOR value]]
        self.aor = aor

        #Init the state structure
        self.feature_structure = feature_structure

    def pick_random_state(self) -> State:
        """
        Select a random state in all the possible state space.
        
        Return a state randomly picked.
        """
        
        #Check if the dict is empty
        if bool(self.feature_structure) == True:
            random_depth: int = np.random.choice(list(self.feature_structure.keys()))
            random_state: State = np.random.choice(self.feature_structure[random_depth])

            return random_state, False
        else:
            return self.start_from_empty_set()

    def start_from_empty_set(self) -> State:
        """
        Start from the empty set (with no feature selected).
        
        Returns the empty initial state.
        """
        
        depth = 0
        if not bool(self.feature_structure):
            return State([0, 0], [], 0, 0.75), True
        else:
            return self.feature_structure[depth][0], True

    def add_to_historic(self, visited_state: State):
        """
        Add to the feature structure historic function.

        visited_state: current state visited by the simulation
        """
        
        state_depth: int = visited_state.number[0]

        #We increment the number of visit of the current state
        if visited_state.description != []:
            visited_state.nb_visited += 1

        if state_depth in self.feature_structure:
            #If there is a key associated to the state depth
            is_already_existing: list = [
                state_searching for state_searching in self.feature_structure[state_depth]
                if visited_state.is_equal(state_searching)
            ]

            if not is_already_existing:
                #If the state is not already registered
                self.feature_structure[state_depth].append(visited_state)
            else:
                #The state already exists
                #We remove the existing state and replace it by the new one
                self.feature_structure[state_depth].remove([
                                                               state_rem for state_rem in
                                                               self.feature_structure[state_depth]
                                                               if visited_state.is_equal(state_rem)
                                                           ][0])
                self.feature_structure[state_depth].append(visited_state)
        else:
            #The layer does not exist, we can simply add it
            self.feature_structure[state_depth] = [visited_state]

    def get_final_aor_sorted(self) -> list:
        """
        Returns the aor table sorted by ascending:

        Index of the feature
        Number of time the feature has been played
        Value of the feature
        Best feature (from the lowest to the biggest)
        """

        index: list = [i for i in range(self.nb_of_features)]
        nb_played: list = self.aor[0]
        values: list = self.aor[1]

        index, values = zip(*sorted(zip(index, values)))

        return [index, nb_played, values, np.argsort(self.aor[1])]
