class Action:
    """
    Action Object.
    """
    
    def __init__(self,
                 state_t,
                 state_next) -> None:
        self.state_t = state_t
        self.state_next = state_next            

    def get_aorf(self, aor_historic: list) -> list:
        """
        Update the ARO of a feature.

        aor_historic: get the not updated aor table.

        Return the AOR table.
        """
        
        # Get the feature played and information about it
        chosen_feature: int = list(set(self.state_next.description)-set(self.state_t.description))[0]

        nb_played: int = aor_historic[0][chosen_feature] + 1    
        aorf_value: float = aor_historic[1][chosen_feature]

        aor_new = aor_historic.copy()
        # Update the value
        aor_new[0][chosen_feature] = nb_played
        aor_new[1][chosen_feature] = ((nb_played-1) * aorf_value + self.state_t.v_value) / nb_played

        return aor_new
