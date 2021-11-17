import jax.numpy as jnp

class CarlaNoConstraints:
    def __init__(self):
        self._state_dim = 5
        self._input_dim = 1

    @property
    def state_dim(self):
        return self._state_dim
    
    @property
    def input_dim(self):
        return self._input_dim

    def f(self, state):
        x, y, v, θ, d = state
        return jnp.array([
            v * jnp.cos(θ), 
            v * jnp.sin(θ),
            -1.06 * v - 0.009 * v ** 2 - 3.6 * d + 3.37,
            0.,
            v - 5.56
        ]).reshape(5, 1)

    def g(self, state):
        x, y, v, θ, d = state
        return jnp.array([
            0., 0., 0., v / 2.51, 0.
        ]).reshape(5, 1)
