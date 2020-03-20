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
            _state : coddiwomple.particles.ParticleState
                state of the system

        """
        self._cumulative_work = 0.
        self._incremental_works = []
        self._proposal_works = []
        self._state = None
        self._ancestry = [index]

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
