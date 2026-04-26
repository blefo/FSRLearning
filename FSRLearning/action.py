from __future__ import annotations

import numpy as np


class Action:
    """
    Transition from state_t to state_next by adding one feature.
    """

    def __init__(self, state_t, state_next) -> None:
        self.state_t = state_t
        self.state_next = state_next

    def get_aorf(self, aor_historic: list, reward_delta: float | None = None) -> list:
        """
        Incrementally update the AOR score for the selected feature.

        AOR_f(k) = ((k - 1) * AOR_f(k-1) + reward_delta) / k
        """
        chosen_feature = list(set(self.state_next.description) - set(self.state_t.description))[0]

        counts = np.array(aor_historic[0], copy=True)
        values = np.array(aor_historic[1], copy=True)

        if reward_delta is None:
            reward_t = 0.0 if self.state_t.reward is None else float(self.state_t.reward)
            reward_next = 0.0 if self.state_next.reward is None else float(self.state_next.reward)
            reward_delta = reward_next - reward_t

        k = int(counts[chosen_feature]) + 1
        prev_aor = float(values[chosen_feature])

        counts[chosen_feature] = k
        values[chosen_feature] = ((k - 1) * prev_aor + float(reward_delta)) / k

        return [counts, values]
