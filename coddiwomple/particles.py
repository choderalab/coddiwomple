"""
Particle Module
"""
#####Imports#####


class Particle(object):
    """
    Generalized Particle Object to hold particle ancestry, logp_proposal_ratios (i.e. logp(K_reverse/K_forward)), logp_state, incremental, and cumulative works
    """
    def __init__(self, index, **kwargs):
        """
        Generalized initialization method for Particle
        arguments
            index : int
                integer of the particle initial ancestry

        parameters
            _cumulative_work : float
                cumulative_work accumulated at time t : -log(weight_(unnormalized,t))
            _incremental_works : list
                incremental (unnormalized) -log(weight_(incremental_unnormalized,t))
            _proposal_works : list
                incremental -log(weight_(proposal,t))
            _state_works : list
                incremental -log(weight_(state, t))
            _state : coddiwomple.particles.ParticleState
                state of the system

        """
        self._cumulative_work = 0.
        self._incremental_works = []
        self._proposal_works = []
        self._state_works = []
        self._state = None
        self._ancestry = [index]

        # provision incremental work
        self._provisional_incremental_work = 0.


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

        def get_last_state_work(self):
            """
            return
                last_state_work : float
                    self._state_works[-1]
            """
            last_state_work = self._state_works[-1]
            return last_state_work

        def set_new_state_work(self, state_work):
            """
            initialize a new state work with the argument value

            arguments
                state_work : float
                    state work increment
            """

            self._state_works.append(state_work)

        def update_proposal_works(self, proposal_work):
            """
            update the proposal_work

            arguments
                proposal_work : float
                    -log(proposal_weight)
            """
            self._proposal_works.append(proposal_work)

        def update_state_work(self, state_work):
            """
            update the state_work in place

            arguments
                state_work : float
                    -log(state_weight)
            """
            self._state_works[-1] = state_work

        def update_ancestry(self, index):
            """
            update the ancestry of the particle

            arguments
                index : int
                    new particle index
            """
            self._ancestry.append(index)

        def update_state(self, state):
            """
            update the state of the particle

            arguments :
                state : coddiwomple.states.ParticleState
            """
            self._state = state


        @property
        def cumulative_work(self):
            return self._cumulative_work
        @property
        def incremental_works(self):
            return self._incremental_works
        @property
        def logp_proposals(self):
            return self._logp_proposals
        @property
        def logp_states(self):
            return self._logp_states
        @property
        def state(self):
            return self._state
        @property
        def ancestry(self):
            return self._ancestry
