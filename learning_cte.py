from haiku.nets import ResNet18
import haiku as hk
import os
import pandas as pd
import optax
import argparse
import jax.numpy as jnp
import jax.random as jrand
import jax
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from core.data.images import loader
from core.utils.meters import AverageMeter
from core.data.load import get_bdy_states_rknn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DIRNAME = 'perception-ckpts-50-epochs'
os.makedirs(DIRNAME, exist_ok=True)

def main():

    rng = jrand.PRNGKey(23)
    train_loader, test_loader = loader(train_bs=64, test_bs=128)
    imgs, _ = next(iter(train_loader))
    
    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.scale(-0.001)
    )
    
    forward = hk.transform_with_state(_forward)
    params, nn_state = forward.init(rng, imgs, is_training=True)
    opt_state = opt_init(params)

    def loss_fn(params, batch, nn_state, is_training=True):
        imgs, labels = batch
        preds, nn_state = forward.apply(params, nn_state, None, imgs, is_training=is_training)
        loss = (1. / preds.shape[0]) * jnp.sum(jnp.square(preds - labels.reshape(preds.shape)))
        return loss, (loss, nn_state)

    @jax.jit
    def update(params, batch, nn_state, opt_state):
        grad, (loss, nn_state) = jax.grad(loss_fn, has_aux=True)(params, batch, nn_state)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state, nn_state
    # We varied it between 50 - 1500 epoch
    for epoch in range(50):
        train_loss_meter = AverageMeter('train_loss')
        test_loss_meter = AverageMeter('test_loss')

        print(f'Beginning epoch {epoch}\n')

        for batch_idx, batch in enumerate(train_loader):
            
            loss, params, opt_state, nn_state = update(params, batch, nn_state, opt_state)
            train_loss_meter.update(loss, n=batch[0].shape[0])

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}\tLoss: {train_loss_meter.avg:.3f}')

        all_preds, all_labels = [], []
        for imgs, labels in tqdm(test_loader):
            preds, _ = forward.apply(params, nn_state, None, imgs, is_training=False)
            test_loss = (1. / preds.shape[0]) * jnp.sum(jnp.square(preds - labels.reshape(preds.shape)))
            # test_loss, _ = loss_fn(params, batch, nn_state, is_training=False)
            test_loss_meter.update(test_loss, n=batch[0].shape[0])
            all_preds.append(jnp.asarray(preds))
            all_labels.append(jnp.asarray(labels.reshape(preds.shape)))

        plot_preds(all_preds, all_labels, test_loader.dataset.states)

        # save checkpoint at the end of the epoch
        save_checkpoint(params, nn_state)

        print(f'\nFinished epoch {epoch}')
        print(f'{"-"*50}')
        print(f'Avg training loss: {train_loss_meter.avg:.4f}\t', end='')
        print(f'Avg test loss: {test_loss_meter.avg:.4f}')
        print(f'{"-"*50}\n')

        # For saving output data into a text file
        with open(f'{DIRNAME}/output_no_crop.txt', 'a') as f:
            output_str = f'\nFinished epoch {epoch}\n'
            output_str += f'{"-" * 50}\n'
            output_str += f'Avg training loss: {train_loss_meter.avg:.4f}\t'
            output_str += f'Avg test loss: {test_loss_meter.avg:.4f}\n'
            output_str += f'{"-"*50}\n'
            # Write output to file
            f.write(output_str)

def _forward(images, is_training):
    return ResNet18(1, resnet_v2=True)(images, is_training)

def save_checkpoint(params, nn_state):
    os.makedirs(DIRNAME, exist_ok=True)
    with open(os.path.join(DIRNAME, 'params.npy'), 'wb') as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(DIRNAME, 'nn_state.npy'), 'wb') as f:
        pickle.dump(nn_state, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_checkpoint():
    with open(os.path.join(DIRNAME, 'params.npy'), 'rb') as f:
        params = pickle.load(f)
    with open(os.path.join(DIRNAME, 'nn_state.npy'), 'rb') as f:
        nn_state = pickle.load(f)

    return params, nn_state

def plot_preds(all_preds, all_labels, state_df):
    preds, labels = np.vstack(all_preds), np.vstack(all_labels)
    df = pd.DataFrame(
        list(zip(np.squeeze(preds), np.squeeze(labels))), 
        columns=['Predictions', 'Labels'])
    full_df = pd.concat([df, state_df], axis=1)
    get_bdy_states_rknn(full_df, ['Predictions', 'speed(m/s)', 'theta_e', 'd'], 200)

    fig, ax = plt.subplots()
    sns.set(style='darkgrid', font_scale=1.5)
    sns.scatterplot(data=full_df, x='Labels', y='Predictions', hue='Safe', ax=ax)
    line = np.linspace(-2, 2, 1000)
    ax.plot(line, line, 'k--')
    ax.set_xlabel(r'True $c_e$')
    ax.set_ylabel(r'Predicted $c_e$')
    handles, labels = ax.get_legend_handles_labels()
    label_dict = {'True': 'Safe', True: 'Safe', 'False': 'Unsafe', False: 'Unsafe'}
    new_labels = [label_dict[label] for label in labels]
    ax.legend(handles=handles[0:], labels=new_labels)
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(f'{DIRNAME}/perception-ckpts.png')
    plt.close()

if __name__ == '__main__':
    main()