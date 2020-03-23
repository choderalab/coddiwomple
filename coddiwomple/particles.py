"""
Particle Module
"""
#####Imports#####


class Particle(object):
    """
    Generalized Particle Object to hold particle ancestry, logp_proposal_ratios (i.e. logp(K_reverse/K_forward)), logp_state, incremental, and cumulative works
    """
    def __init__(self, index, record_states = False, **kwargs):
        """
        Generalized initialization method for Particle
        arguments
            index : int
                integer of the particle initial ancestry
            record_states : bool, default False
                whether to record the ParticleState history

        parameters
            _cumulative_work : float
                cumulative_work accumulated at time t : -log(weight_(unnormalized,t))
            _incremental_works : list
                incremental (unnormalized) -log(weight_(incremental_unnormalized,t))
            _proposal_works : list
                incremental -log(weight_(proposal,t))
            _state : coddiwomple.particles.ParticleState
                state of the system
            _record_states : bool, default False
                whether to record the ParticleState history

        """
        self._cumulative_work = 0.
        self._incremental_works = []
        self._proposal_works = []
        self._state = None
        self._ancestry = [index]

        self._record_states = record_states
        if self._record_states:
            self._state_history = []
        else:
            self._state_history = None

        # provision incremental work
        self._auxiliary_work = 0.

        def update_auxiliary_work(self, incremental_work):
            """
            update the auxiliary work in place

            arguments
                incremental_work : float
            """
            self._auxiliary_work += incremental_work

        def zero_auxiliary_work(self):
            """
            reset the auxiliary work to 0.
            """
            self._auxiliary_work = 0.

        def update_work(self, incremental_work):
            """
            update the cumulative work and log the increment

            arguments
                incremental_work : float
                    incremental work
                reset_provisional_work : bool, default True
                    whether to reset the _provisional_incremental_work
            """
            self._cumulative_work += incremental_work
            self._incremental_works.append(incremental_work)

        def get_last_proposal_work(self):
            """
            return
                last_proposal_work : float
                    self._proposal_works[-1]
            """
            last_proposal_work = self._proposal_works[-1]
            return last_proposal_work

        def update_proposal_work(self, proposal_work):
            """
            update the proposal_work

            arguments
                proposal_work : float
                    -log(proposal_weight)
            """
            self._proposal_works.append(proposal_work)

        def update_proposal_work_in_place(self, proposal_work):
            """
            update the proposal_work in place

            arguments
                proposal_work : float
                    -log(proposal_weight)
            """
            self._proposal_works[-1] = proposal_work

        def update_ancestry(self, index):
            """
            update the ancestry of the particle

            arguments
                index : int
                    new particle index
            """
            self._ancestry.append(index)

        def update_state(self, state, amend_state_history = False, state_history_index = -1):
            """
            update the state of the particle

            arguments :
                state : coddiwomple.states.ParticleState
                    updated state
                amend_state_history : bool, default False
                    whether to modify the state history
                state_history_index : int
                    index of the state history to update

            NOTE : if self._record_states is False, amend_state_history and state_history_index are not called, nor is a state history update conducted
            """
            if self._record_states:
                if amend_state_history:
                    self._state_history[state_history_index] = copy.deepcopy(state)
                else:
                    self._state_history.append(copy.deepcopy(state))

            self._state = state



        @property
        def cumulative_work(self):
            return self._cumulative_work
        @property
        def incremental_works(self):
            return self._incremental_works
        @property
        def proposal_works(self):
            return self._proposal_works
        @property
        def state(self):
            return self._state
        @property
        def ancestry(self):
            return self._ancestry
        @property
        def auxiliary_work(self):
            return self._auxiliary_work
        @property
        def state_history(self):
            return self._state_history
