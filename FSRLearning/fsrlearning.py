from __future__ import annotations

from collections import defaultdict
import numpy as np

from .state import State


class FeatureSelectionProcess:
    """
    Holds graph memory for the feature-selection RL process.
    """

    def __init__(
        self,
        nb_of_features: int,
        eps: float,
        alpha: float,
        gamma: float,
        aor: list,
        feature_structure: dict,
        random_state: int | None = None,
    ) -> None:
        self.nb_of_features = nb_of_features
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

        self.aor = aor
        self.feature_structure = feature_structure

        self.rng = np.random.default_rng(random_state)

        self._state_map: dict[frozenset, State] = {}
        self._depth_index: defaultdict[int, int] = defaultdict(int)

        # Per-state action repository used by exploit phases
        # key: state_key -> {feature_idx -> {'count': int, 'mean': float}}
        self.state_action_repository: dict[frozenset, dict[int, dict[str, float]]] = {}

        # Create and register the empty state at depth 0.
        empty_state = State([0, 0], [], 0.0, 0.0, 0)
        self._register_state(empty_state)

    def _register_state(self, state: State) -> State:
        key = state.key
        if key in self._state_map:
            return self._state_map[key]

        depth = len(state.description)
        state.number = [depth, self._depth_index[depth]]
        self._depth_index[depth] += 1

        self._state_map[key] = state
        if depth not in self.feature_structure:
            self.feature_structure[depth] = []
        self.feature_structure[depth].append(state)
        self.state_action_repository[key] = {}
        return state

    def get_or_create_state(self, description: list[int]) -> State:
        key = frozenset(description)
        if key in self._state_map:
            return self._state_map[key]

        new_state = State([len(description), 0], sorted(description), 0.0, None, 0)
        return self._register_state(new_state)

    def pick_random_state(self) -> tuple[State, bool]:
        """
        Sample an episode start state from the full state space (excluding terminal state).
        """
        if self.nb_of_features <= 1:
            return self.start_from_empty_set()

        k = int(self.rng.integers(0, self.nb_of_features))
        if k == 0:
            return self.start_from_empty_set()

        sampled = self.rng.choice(np.arange(self.nb_of_features), size=k, replace=False)
        random_state = self.get_or_create_state(sampled.tolist())
        return random_state, False

    def start_from_empty_set(self) -> tuple[State, bool]:
        return self.get_or_create_state([]), True

    def add_to_historic(self, visited_state: State) -> None:
        """
        Increment visit count and keep state in the graph index.
        """
        registered = self.get_or_create_state(visited_state.description)
        registered.nb_visited += 1

        # Keep runtime-updated values synchronized into the canonical object.
        registered.v_value = visited_state.v_value
        registered.reward = visited_state.reward

    def get_available_features(self, state: State) -> list[int]:
        selected = set(state.description)
        return [f for f in range(self.nb_of_features) if f not in selected]

    def update_state_action_repository(self, state: State, feature: int, reward_delta: float) -> None:
        repo = self.state_action_repository[state.key]
        if feature not in repo:
            repo[feature] = {"count": 1, "mean": float(reward_delta)}
            return

        prev_count = int(repo[feature]["count"])
        prev_mean = float(repo[feature]["mean"])
        new_count = prev_count + 1
        new_mean = ((prev_count * prev_mean) + float(reward_delta)) / new_count
        repo[feature]["count"] = new_count
        repo[feature]["mean"] = new_mean

    def get_final_aor_sorted(self) -> list:
        index = np.arange(self.nb_of_features)
        nb_played = np.array(self.aor[0], copy=True)
        values = np.array(self.aor[1], copy=True)

        sorted_idx = np.argsort(values)
        return [tuple(index.tolist()), nb_played, tuple(values.tolist()), sorted_idx]
