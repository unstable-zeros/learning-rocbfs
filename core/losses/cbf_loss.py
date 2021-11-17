import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.flatten_util import ravel_pytree
from functools import partial

class CBFLoss:

    def __init__(self, hparams, network, dynamics, alpha, dual_vars):
        self._hparams = hparams
        self._network = network
        self._dynamics = dynamics
        self._alpha = alpha
        self._dual_vars = dual_vars

    @partial(jax.jit, static_argnums=0)
    def cbf_term(self, params, states, disturbances):
        """Computes the LHS of the CBF inequality term with sup over the l_inf ball.
        
            sup_{||u||_inf <= 1} <dh(x), f(x) + g(x)u> + alpha(h(x)) 
                = <dh(x), f(x)> + ||dh(x).T g(x)||_1 + alpha(h(x))
            
        If self._hparams.robust True, we subtract the robustness term
        
            - lambda_robust * ||dh(x)||_2

        from the CBF LHS expression.

        Args:
            params: parameters of neural network h(x).
            states: states on which we evaluate the CBF inequality LHS.

        Returns:
            LHS of CBF inequality with optional robustness term.
        """

        def cbf_term_indiv(data):

            x, d = data[:4], data[4]
    
            scalar_network = lambda x_ : jnp.sum(self._network.apply(params, x_))
            dh = jax.grad(scalar_network)(x)

            term1 = jnp.dot(dh, self._dynamics.f(x, d))
            term2 = jnp.linalg.norm(jnp.dot(dh.T, self._dynamics.g(x)), ord=1)
            term3 = self._alpha(scalar_network(x))
            cbf_term_ = term1 + term2 + term3

            if self._hparams.robust is True:
                cbf_term_ -= self._hparams.lambda_robust * jnp.linalg.norm(dh)

            if self._hparams.use_lip_output_term is True:
                cbf_term_ -= self._hparams.lip_output_term

            return cbf_term_

        full_data = jnp.hstack((states, disturbances))
        return jax.vmap(cbf_term_indiv, in_axes=(0))(full_data)

    # @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, data_dict):
        loss, _, _, _ = self.__loss(params, data_dict)
        return loss

    def constraints_fn(self, params, data_dict):
        _, constraints, _, _ = self.__loss(params, data_dict)
        return {key: float(value) for (key, value) in constraints.items()}

    def diffs_fn(self, params, data_dict):
        _, _, diffs, losses = self.__loss(params, data_dict)

        return diffs

    def all_losses_fn(self, params, data_dict):
        _, _, _, losses = self.__loss(params, data_dict)
        return {key: float(value) for (key, value) in losses.items()}

    @staticmethod
    def loss_with_dual_var(dual_var, diff):
        if dual_var.shape == ():
            return dual_var * jnp.sum(jnn.relu(diff))
        return jnp.dot(dual_var, jnn.relu(diff))

    @staticmethod
    def const_satisfaction(const):
        frac_incorrect = jnp.sum(jnp.heaviside(const, 0)) / const.shape[0]
        return (1.0 - frac_incorrect) * 100.

    # @partial(jax.jit, static_argnums=0)
    def __loss(self, params, data_dict):            

        consts = dict.fromkeys(['safe', 'unsafe', 'dyn'])
        diffs = dict.fromkeys(['safe', 'unsafe', 'dyn'])
        losses = dict.fromkeys(['safe', 'unsafe', 'dyn', 'param'])

        # h(x_safe) >= \gamma_safe <=> \gamma_safe - h(x_safe) <= 0
        safe_output = self._network.apply(params, data_dict['safe'])
        diffs['safe'] = self._hparams.gamma_safe - safe_output
        losses['safe'] = self.loss_with_dual_var(self._dual_vars['safe'], diffs['safe'])
        consts['safe'] = self.const_satisfaction(diffs['safe'])

        # h(x_unsafe) <= -\gamma_unsafe <=> \gamma_unsafe + h(x_unsafe) <= 0
        unsafe_output = self._network.apply(params, data_dict['unsafe'])
        diffs['unsafe'] = self._hparams.gamma_unsafe + unsafe_output
        losses['unsafe'] = self.loss_with_dual_var(self._dual_vars['unsafe'], diffs['unsafe'])
        consts['unsafe'] = self.const_satisfaction(diffs['unsafe'])

        # q(u, x) >= \gamma_dyn <=> \gamma_dyn - q(u, x) <= 0
        cbf_output = self.cbf_term(params, data_dict['all'], data_dict['all_dists'])
        diffs['dyn'] = self._hparams.gamma_dyn - cbf_output
        losses['dyn'] = self.loss_with_dual_var(self._dual_vars['dyn'], diffs['dyn'])
        consts['dyn'] = self.const_satisfaction(diffs['dyn'])

        # Penalize large values of the derivative
        def grad_indiv(x):
            scalar_network = lambda x_ : jnp.sum(self._network.apply(params, x_))
            return jax.grad(scalar_network)(x)
            
        grad_all_output = jax.vmap(grad_indiv, in_axes=(0))(data_dict['all'])
        losses['grad'] = self._hparams.lambda_grad * jnp.sum(jnp.square(grad_all_output))

        # Penalize large parameter values
        losses['param'] = self._hparams.lambda_param * jnp.sum(jnp.square(ravel_pytree(params)[0]))

        # loss = safe_loss + unsafe_loss + dyn_loss + dh_loss + param_loss
        loss = losses['safe'] + losses['unsafe'] + losses['dyn'] + losses['param'] + losses['grad']

        return jnp.reshape(loss, ()), consts, diffs, losses