import pandas as pd
import haiku as hk
import jax.numpy as jnp
import optax
import jax
import jax.nn as jnn
import wandb
import os
import pickle
import matplotlib.pyplot as plt

from core.utils.parse_args import parse_args
from core.utils.viz import Visualizer
from core.data.load import load_data_v2
from core.losses.new_cbf_loss import CBFLoss
from core.output_maps.pos_to_velocity import PosToVelocity
from core.output_maps.img_to_cte import ImgToCTE
from core.dynamics.carla_4state import CarlaDynamics
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    config = wandb.config
    config.n_epochs = args.n_epochs
    config.learning_rate = args.learning_rate
    config.dual_step_size = args.dual_step_size
    key_seq = hk.PRNGSequence(23)
    
    # output_map = PosToVelocity()
    output_map = ImgToCTE()

    data_dict, T_x = load_data_v2(args, output_map=output_map)
    alpha = lambda x : x
    dual_vars = init_dual_vars(args, data_dict)

    net = hk.without_apply_rng(hk.transform(lambda x: net_fn(args)(x)))
    dynamics = CarlaDynamics(T_x)

    PrimalDualLoss = CBFLoss(args, net, dynamics, alpha, dual_vars, T_x)
    viz = Visualizer(net, args.results_path, data_dict, PrimalDualLoss.cbf_term)

    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.scale(-args.learning_rate)
    )

    params = net.init(next(key_seq), jnp.zeros([1, dynamics.state_dim]))
    opt_state = opt_init(params)
    
    loss = PrimalDualLoss.loss_fn(params, data_dict)
    losses_over_steps = []
    steps = []

    for step in range(args.n_epochs):

        # primal step
        loss, grad = jax.value_and_grad(PrimalDualLoss.loss_fn)(params, data_dict)
        
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        # dual step
        diffs = PrimalDualLoss.diffs_fn(params, data_dict)
        losses_over_steps.append(loss.item())
        steps.append(step)

        for key in dual_vars.keys():
            dv = dual_vars[key]
            if args.dual_scheme == 'avg':
                dual_vars[key] = jnn.relu(dv + args.dual_step_size * jnp.sum(diffs[key]))
            else:
                dual_vars[key] = jnn.relu(dv + args.dual_step_size * diffs[key].squeeze())

        if step % 100 == 0:
            print(f'Step: {step}/{args.n_epochs}')
            consts = PrimalDualLoss.constraints_fn(params, data_dict)
            for key in consts.keys():
                print(f'{key} pct: {consts[key]:.3f}\t', end='')
            wandb.log({f'{key} constraint': consts[key] for key in consts.keys()})
            print('')

            print(f'Loss: {loss:.3f}\t', end='')
            wandb.log({f'Total loss': loss.item()})
            losses = PrimalDualLoss.all_losses_fn(params, data_dict)
            for key in losses.keys():
                print(f'{key} loss: {losses[key]:.3f}\t', end='')
            wandb.log({f'{key} loss': losses[key] for key in losses.keys()})
            print('\n')

            if args.dual_scheme == 'ae':
                wandb.log({f'{key} dual var': jnp.linalg.norm(dual_vars[key], ord=2).item() for key in dual_vars.keys()})
            else:
                wandb.log({f'{key} dual var': dual_vars[key].item() for key in dual_vars.keys()})

            viz.state_separation(params)
            viz.single_level_set(params)
            viz.level_sets(params)

            fname = os.path.join(args.results_path, 'trained_cbf.npy')
            with open(fname, 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        df_losses = pd.DataFrame({f'{key}_loss': [losses[key]] for key in losses.keys()})
        df_losses.to_csv(os.path.join(args.results_path, f'losses_.txt'), sep='\t', index=False)
        plt.figure(figsize=(12, 6))  # Set the figure size to a larger size
        plt.plot(steps, losses_over_steps)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title(f'Loss over Steps')
        plt.xticks(range(0, args.n_epochs + 1, 100), rotation='vertical')  # Add more x-axis ticks
        plt.savefig(os.path.join(args.results_path, f'loss_plot_.png'))
        plt.close()

    wandb.finish()

def net_fn(args):
    """Feed-forward NN architecture."""

    layers = []
    for dim in args.net_dims:
        layers.extend([hk.Linear(dim), jnp.tanh])
    layers.append(hk.Linear(1))

    return hk.Sequential(layers)

def init_dual_vars(args, data_dict):

    if args.dual_scheme == 'avg':
        return {'safe': jnp.array(1.0), 'unsafe': jnp.array(1.0), 'dyn': jnp.array(1.0)}
    elif args.dual_scheme == 'ae':
        return {
            'safe': jnp.ones(data_dict['safe'].shape[0]), 
            'unsafe': jnp.ones(data_dict['unsafe'].shape[0]),
            'dyn': jnp.ones(data_dict['all'].shape[0])
        }
    else:
        raise ValueError(f'Dual scheme {args.dual_scheme} is not supported.')

if __name__ == '__main__':
    args = parse_args()
    main(args)