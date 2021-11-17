import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.neighbors import KDTree
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import os

from viz import plot_peanut, double_grid

def load_data_v2(args, output_map=None):

    # dictionary for storing meta data
    meta_data = {}

    # load data from file
    dfs = [pd.read_pickle(path) for path in args.data_path]
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

    # clip some outliers
    df = df[df['theta_e'] <= 1.0]

    if output_map is None:
        state_cols = ['cte', 'speed(m/s)', 'theta_e', 'd']
    else:
        state_cols = output_map.state_cols
        df = output_map.map(df)

    disturbance_cols, input_cols = ['dphi_t'], ['input']
    all_cols = state_cols + disturbance_cols + input_cols

    if args.normalize_state is True:
        max_cte = df['cte'].abs().max()
        max_speed = df['speed(m/s)'].abs().max()
        max_theta_e = df['theta_e'].abs().max()
        max_d = df['d'].abs().max()

        meta_data['normalizers'] = {
            'cte': max_cte, 'speed': max_speed, 'theta_e': max_theta_e, 'd': max_d
        }

        df['cte'] = df['cte'] / max_cte
        df['speed(m/s)'] = df['speed(m/s)'] / max_speed
        df['theta_e'] = df['theta_e'] / max_theta_e
        df['d'] = df['d'] / max_d

        # T_x should normalize the state
        T_x = jnp.diag(jnp.array([
            1. / max_cte, 1. / max_speed, 1. / max_theta_e, 1. / max_d
        ]))
    else:
        T_x = jnp.eye(4)

    df = df[all_cols]

    if args.data_augmentation is True:
        df_copy = df.copy()
        df_copy['cte'] = df_copy['cte'].multiply(-1)
        df_copy['theta_e'] = df_copy['theta_e'].multiply(-1)
        df_copy['dphi_t'] = df_copy['dphi_t'].multiply(-1)
        df = pd.concat([df, df_copy], ignore_index=True)

    n_all = len(df.index)
    # get_bdy_states_v2(df, state_cols, args.nbr_thresh, args.min_n_nbrs)
    get_bdy_states_rknn(df, state_cols, 200)

    if args.n_samp_all != 0:
        df_copy = df.copy()
        df[state_cols] += 0.01 * np.random.randn(*df[state_cols].shape)
        df = pd.concat([df, df_copy])
    
    n_safe, n_unsafe = len(df[df.Safe == 1].index), len(df[df.Safe == 0].index)

    data_dict = {
        'safe': df[df.Safe == 1][state_cols].to_numpy(),
        'unsafe': df[df.Safe == 0][state_cols].to_numpy(),
        'all': df[state_cols].to_numpy(),
        'all_dists': df[disturbance_cols].to_numpy(),
        'all_inputs': df[input_cols].to_numpy()
    }

    create_tables(n_all, n_safe, n_unsafe, args, meta_data)
    _save_meta_data(meta_data, args)

    # for var_to_bin in state_cols:
    #     plot_peanut(df[state_cols + ['Safe']], state_cols, var_to_bin=var_to_bin, include_safe=True)
    #     plot_peanut(df[state_cols + ['Safe']], state_cols, var_to_bin=var_to_bin, include_safe=False)
    # quit()
    # plot_peanut(df[state_cols + ['Safe']], state_cols, var_to_bin='speed(m/s)', include_safe=True)
    # plot_peanut(df[state_cols + ['Safe']], state_cols, var_to_bin='speed(m/s)', include_safe=False)

    # double_grid(df)

    return data_dict, T_x



def get_bdy_states_rknn(df, state_cols, k):
    """Implementation of RkNN for identifying boundary points."""

    # this is the data matrix -- should be [N, STATE_DIM]
    state_matrix = df[state_cols].to_numpy()

    # great a KD-tree object
    tree = KDTree(state_matrix)

    # query the KD-tree for a particular value of k.  this should give 
    # a matrix of size [N, k], where the i^th row has the indicies of
    # the k nearest neighbors to the i^th data point.
    _, knn_inds = tree.query(state_matrix, k=k)

    # flatten the indices.  This will give us a list of all nodes
    # that are deemed to be a neighbor of some other node.  This list
    # should be of size [1, k * N]
    flat_inds = knn_inds.flatten()

    # now we count the number of occurences of each number in the 
    # flattened array.  This should give us a list of size [1, N]
    counts = np.bincount(flat_inds)

    pct_unsafe = 0.4
    nbr_thresh = np.quantile(counts, pct_unsafe)

    # threshold the counts 
    df['Safe'] = np.array(counts >= nbr_thresh)


def get_bdy_states_v2(df, state_cols, thresh, min_num_nbrs):
    state_matrix = df[state_cols].to_numpy()
    tree = KDTree(state_matrix)
    dists = tree.query_radius(state_matrix, r=thresh, count_only=True)
    df['Safe'] = np.array(dists < min_num_nbrs)


def sample_extra_states(states, num_samp):
    """Samples extra states in neighborhoods of given states.
    
    Args:
        states: [N, STATE_DIM] matrix containing states.
        num_samp: Number of additional states to sample for each state in {states}.

    Returns:
        [N * n_samp, STATE_DIM] matrix of states.
    """

    extra_states = [states + 0.01 * np.random.randn(*states.shape) for _ in range(num_samp)]
    return np.vstack(extra_states)

def create_tables(n_all, n_safe, n_unsafe, args, meta_data):

    meta_data['pct_safe'] = (n_safe * (args.n_samp_safe + 1)) / (n_all * (args.n_samp_all + 1)) * 100
    meta_data['pct_unsafe'] = (n_unsafe * (args.n_samp_unsafe + 1)) / (n_all * (args.n_samp_all + 1)) * 100
    meta_data['num-expert-states'] = {'all': n_all, 'safe': n_safe, 'unsafe': n_unsafe}
    meta_data['num-samp-states'] = {
        'all': n_all * args.n_samp_all, 'safe': n_safe * args.n_samp_safe, 'unsafe': n_unsafe * args.n_samp_unsafe
    }
    meta_data['num-total-states'] = {
        'all': n_all * (args.n_samp_all + 1), 'safe': n_safe * (args.n_samp_safe + 1), 'unsafe': n_unsafe * (args.n_samp_unsafe + 1)
    }

    expert_table = PrettyTable()
    expert_table.align = 'l'
    expert_table.field_names = ['State type', '# expert states', '# sampled states', 'Total', 'Percent of Total']
    expert_table.add_row(['All', n_all, n_all * args.n_samp_all, n_all * (args.n_samp_all + 1), '--'])
    expert_table.add_row(['Safe', n_safe, n_safe * args.n_samp_safe, n_safe * (args.n_samp_safe + 1), f'{meta_data["pct_safe"]:.1f} %'])
    expert_table.add_row(['Unsafe', n_unsafe, n_unsafe * args.n_samp_unsafe, n_unsafe * (args.n_samp_unsafe + 1), f'{meta_data["pct_unsafe"]:.1f} %'])
    print(expert_table)

def _save_meta_data(meta_data, args):
    """Saves meta data line arguments to JSON file.
    
    Args:
        d: Dictionary of items.
        args: Command line arguments
    """

    fname = os.path.join(args.results_path, 'meta_data.json')
    with open(fname, 'w') as f:
        json.dump(meta_data, f, indent=2)