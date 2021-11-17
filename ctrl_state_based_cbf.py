import cvxpy as cp
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import json
import os

from core.dynamics.carla_4state import CarlaDynamics

ROOT = '../old_trained_results/0720/results-less-robust-low-margins-all-data/'
CBF_PATH = os.path.join(ROOT, 'trained_cbf.npy')
ARGS_PATH = os.path.join(ROOT, 'args.json')
META_DATA_PATH = os.path.join(ROOT, 'meta_data.json')


def main():

    args_dict = load_json(ARGS_PATH)
    meta_data = load_json(META_DATA_PATH)

    net = hk.without_apply_rng(hk.transform(lambda x: net_fn()(x)))
    with open(CBF_PATH, 'rb') as handle:
        loaded_params = pickle.load(handle)

    def learned_h(x): return jnp.sum(net.apply(loaded_params, x))
    zero_ctrl = get_zero_controller()

    safe_ctrl = make_safe_controller(zero_ctrl, learned_h, args_dict, meta_data)

    # Example usage:
    u = safe_ctrl(jnp.array([1., 2., 3., 4.]), 1)


def make_safe_controller(nominal_ctrl, h, args_dict, meta_data):
    """Create a safe controller using learned hybrid CBF."""

    delta_f, delta_g = args_dict['delta_f'], args_dict['delta_g']
    # lip_const_a, lip_const_b = args_dict['lip_const_a'], args_dict['lip_const_b']
    use_output_map = False
    # use_output_map = args_dict['use_lip_output_term']

    T_x = jnp.eye(4)

    dh = jax.grad(h, argnums=0)
    dyn = CarlaDynamics(T_x)
    def alpha(x): return x
    def norm(x): return jnp.linalg.norm(x)
    def cpnorm(x): return cp.norm(x)
    def dot(x, y): return jnp.dot(x, y)

    def safe_ctrl(x, d):
        """Solves HCBF-QP to map an input state to a safe action u.

        Params:
            x: state.
            d: disturbance.
        """

        cte, v, θ_e, d_var = x

        # compute action used by nominal controller
        u_nom = nominal_ctrl(x)

        # setup and solve HCBF-QP with CVXPY
        u_mod = cp.Variable(len(u_nom))
        obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))

        if use_output_map is False:
            constraints = [
                dot(dh(x), dyn.f(x, d)) + u_mod.T @ dot(dyn.g(x).T, dh(x)) +
                alpha(h(x)) - norm(dh(x)) * (delta_f + delta_g * cpnorm(u_mod)) >= 0
            ]
        else:
            constraints = [
                dot(dh(x), dyn.f(x, d)) + u_mod.T @ dot(dyn.g(x).T, dh(x)) + alpha(h(x)) - norm(dh(x)) *
                (delta_f + delta_g * cpnorm(u_mod)) - (lip_const_a + lip_const_b * cpnorm(u_mod)) >= 0
            ]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-10)

        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return u_mod.value, h(x), dh(x), (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        return jnp.array([0.]), h(x), dh(x), (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))

    return safe_ctrl


def net_fn(net_dims=[32, 16]):
    """Feed-forward NN architecture."""

    layers = []
    for dim in net_dims:
        layers.extend([hk.Linear(dim), jnp.tanh])
    layers.append(hk.Linear(1))

    return hk.Sequential(layers)


def get_zero_controller():
    """Returns a zero controller"""

    return lambda state: jnp.array([0.])


def load_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return data


if __name__ == '__main__':
    main()
