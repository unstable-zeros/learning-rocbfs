import jax.numpy as jnp

class CarlaDynamics:
    def __init__(self, T_x):
        self._state_dim = 4
        self._input_dim = 1
        self._Tx = T_x
        self._inv_Tx = jnp.linalg.inv(self._Tx)

    @property
    def state_dim(self):
        return self._state_dim
    
    @property
    def input_dim(self):
        return self._input_dim

    def f(self, state, disturbance):
        unnormalized_state = self._inv_Tx @ state.reshape(self._state_dim, 1)
        c, v, θ_e, d = unnormalized_state.reshape(self._state_dim,)
        ϕ_dot_t = disturbance
        new_state = jnp.array([
            v * jnp.sin(θ_e), 
            -1.0954 * v - 0.007 * v ** 2 - 0.1521 * d + 3.37387,
            -ϕ_dot_t,
            3.6 * v - 20
        ]).reshape(self._state_dim, 1)
        return self._Tx @ new_state

    def g(self, state):
        unnormalized_state = self._inv_Tx @ state.reshape(self._state_dim, 1)
        c, v, θ, d = unnormalized_state.reshape(self._state_dim,)
        new_state = jnp.array([
            0., 0., v / 2.51, 0.
        ]).reshape(self._state_dim, 1)
        return self._Tx @ new_state