import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools

N_CHUNKS = 5
N_ROWS = 2
N_COLS = 3

def plot_peanut(df, state_cols, var_to_bin='d', include_safe=True):

    non_bin_cols = state_cols.copy()
    non_bin_cols.remove(var_to_bin)

    min_d, max_d = df[var_to_bin].min(), df[var_to_bin].max()
    print(f'Max var: {max_d} | Min var: {min_d}')
    bins = list(np.linspace(min_d, max_d, num=N_CHUNKS+1, endpoint=True))
    print(bins)

    get_chunk = lambda row, bins : np.digitize(row[var_to_bin], bins)
    df['Chunk'] = df.apply(lambda row: get_chunk(row, bins), axis=1)

    sns.set(style='white', font_scale=1.6)
    fig, axes = plt.subplots(nrows=1, ncols=N_CHUNKS, subplot_kw={'projection': '3d'}, figsize=(30,30))

    for ch_idx in range(N_CHUNKS):

        ax = axes[ch_idx]
        curr_df = df[df['Chunk'] == ch_idx+1]
        safe_data = curr_df[curr_df.Safe == True][non_bin_cols].to_numpy()
        unsafe_data = curr_df[curr_df.Safe == False][non_bin_cols].to_numpy()
        
        if include_safe is True:
            ax.scatter(safe_data[:, 0], safe_data[:, 1], safe_data[:, 2], color='blue', label='Safe')
        ax.scatter(unsafe_data[:, 0], unsafe_data[:, 1], unsafe_data[:, 2], color='orange', label='Unsafe')

        ax.set_title(f'{bins[ch_idx]:.2f} $\leq$ {var_to_bin} $\leq$ {bins[ch_idx+1]:.2f}')
        print(f'Chunk: {bins[ch_idx]:.2f} <= {var_to_bin} <= {bins[ch_idx+1]:.2f}', end='')

    plt.legend()
    if 'speed' in var_to_bin:
        var_to_bin = 'speed' 
    plt.savefig(f'peanut-binned-{var_to_bin}-include-safe-{include_safe}.png')



def double_grid(df):

    # bin v and d
    min_d, max_d = df['d'].min(), df['d'].max()
    min_v, max_v = df['speed(m/s)'].min(), df['speed(m/s)'].max()

    bins_d = list(np.linspace(min_d, max_d, num=N_CHUNKS+1, endpoint=True))
    bins_v = list(np.linspace(min_v, max_v, num=N_CHUNKS+1, endpoint=True))

    get_d_chunk = lambda row, bins : np.digitize(row['d'], bins)
    get_v_chunk = lambda row, bins : np.digitize(row['speed(m/s)'], bins)
    
    df['Chunk-d'] = df.apply(lambda row: get_d_chunk(row, bins_d), axis=1)
    df['Chunk-v'] = df.apply(lambda row: get_v_chunk(row, bins_v), axis=1)

    fig, axes = plt.subplots(nrows=N_CHUNKS, ncols=N_CHUNKS, figsize=(30,30))

    for d_idx in range(N_CHUNKS):
        for v_idx in range(N_CHUNKS):

            ax = axes[d_idx, v_idx]
            curr_df = df[(df['Chunk-d'] == d_idx + 1) & (df['Chunk-v'] == v_idx + 1)]

            safe_data = curr_df[curr_df.Safe == True][['cte', 'speed(m/s)']].to_numpy()
            unsafe_data = curr_df[curr_df.Safe == False][['cte', 'speed(m/s)']].to_numpy()
            
            # ax.scatter(safe_data[:, 0], safe_data[:, 1], color='blue', label='Safe')
            ax.scatter(unsafe_data[:, 0], unsafe_data[:, 1], color='orange', label='Unsafe')
            # sns.scatterplot(ax=ax, data=curr_df, hue='Safe', x='theta_e', y='cte')

            name = f'{bins_d[d_idx]:.2f} $\leq$ d $\leq$ {bins_d[d_idx+1]:.5f} and {bins_v[v_idx]:.5f} $\leq$ v $\leq$ {bins_v[v_idx+1]:.2f}'
            ax.set_title(name)
            ax.set_xlabel('theta_e')
            ax.set_ylabel('cte')
            print(f'Chunk: {name}')

    plt.tight_layout()
    plt.savefig('peanut-grid.png')