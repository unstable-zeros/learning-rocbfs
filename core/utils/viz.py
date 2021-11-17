import seaborn as sns
import pandas as pd
import os 
import matplotlib.pyplot as plt
import wandb
import numpy as np
import jax 
import jax.numpy as jnp

sns.set(style='darkgrid', font_scale=2.5)

class Visualizer:
    def __init__(self, net, results_path, data_dict, cbf_fn):
        self._net = net
        self._results_path = results_path
        self._data_dict = data_dict
        self._cbf_fn = cbf_fn

        safe_df = pd.DataFrame(self._data_dict['safe'], columns=['cte', 'speed', 'theta_e', 'd'])
        safe_df['Safe'] = True

        unsafe_df = pd.DataFrame(self._data_dict['unsafe'], columns=['cte', 'speed', 'theta_e', 'd'])
        unsafe_df['Safe'] = False

        self._full_df = pd.concat([safe_df, unsafe_df], ignore_index=True)

        self._chunk_df()

        
    def _chunk_df(self):

        n_chunks = 3

        def get_bins(var):
            min_val, max_val = self._full_df[var].min(), self._full_df[var].max()
            return list(np.linspace(min_val, max_val, num=n_chunks + 1, endpoint=True))

        self._bins_d, self._bins_v = get_bins('d'), get_bins('speed')
        get_d_chunk = lambda row, bins : np.digitize(row['d'], bins)
        get_v_chunk = lambda row, bins : np.digitize(row['speed'], bins)

        self._full_df['Chunk-d'] = self._full_df.apply(
                            lambda row: get_d_chunk(row, self._bins_d), axis=1)
        self._full_df['Chunk-speed'] = self._full_df.apply(
                                lambda row: get_v_chunk(row, self._bins_v), axis=1)

    def state_separation(self, params):

        def make_df(safe):
            output = self._net.apply(params, self._data_dict[safe])
            df = pd.DataFrame(output, columns=['h(x)'])
            df['Constraint-Type'] = safe.capitalize()
            return df

        safe_df, unsafe_df = make_df(safe='safe'), make_df(safe='unsafe')
        df = pd.concat([safe_df, unsafe_df], ignore_index=True)
        
        plt.figure()
        sns.boxplot(data=df, x='Constraint-Type', y='h(x)')
        wandb.log({'separation': wandb.Image(plt)})
        plt.savefig(os.path.join(self._results_path, 'state_separation.png'))
        plt.close()

    def single_level_set(self, params):

        df = self._full_df
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))


        safe_df = pd.DataFrame(self._data_dict['safe'], columns=['cte', 'speed', 'theta_e', 'd'])
        safe_df['Safe'] = True
        unsafe_df = pd.DataFrame(self._data_dict['unsafe'], columns=['cte', 'speed', 'theta_e', 'd'])
        unsafe_df['Safe'] = False
        all_df = pd.concat([safe_df, unsafe_df], ignore_index=True)

        sns.scatterplot(data=all_df, ax=ax, x='cte', y='theta_e') #, hue='Safe')

        min_cte, max_cte = df['cte'].min(), df['cte'].max()
        min_theta_e, max_theta_e = df['theta_e'].min(), df['theta_e'].max()

        cte_range = np.linspace(min_cte, max_cte, num=200)
        theta_e_range = np.linspace(min_theta_e, max_theta_e, num=200)
        avg_d = df['d'].mean()
        avg_speed = df['speed'].mean()

        hvals = jax.vmap(
            lambda s1: jax.vmap(
                lambda s2: self._net.apply(params, jnp.array([s1, avg_speed, s2, avg_d]))
            )(theta_e_range)
        )(cte_range).squeeze()

        cntr_plt = ax.contour(cte_range, theta_e_range, hvals, linewidths=3, colors='k', levels=4)
        plt.clabel(cntr_plt, inline=1, fontsize=25)
        ax.set_xlabel(r'$c_e$')
        ax.set_ylabel(r'$\theta_e$')
        handles, labels = ax.get_legend_handles_labels()
        label_dict = {'True': 'Safe', True: 'Safe', 'False': 'Unsafe', False: 'Unsafe'}
        new_labels = [label_dict[label] for label in labels]
        ax.legend(handles=handles[0:], labels=new_labels)
        plt.subplots_adjust(left=0.2)

        wandb.log({'level_set': wandb.Image(plt)})
        plt.savefig(os.path.join(self._results_path, 'level_set.png'))
        plt.close()



    def level_sets(self, params):

        n_chunks = 3
        df = self._full_df
        fig, axes = plt.subplots(nrows=n_chunks, ncols=n_chunks, figsize=(20,20))

        min_cte, max_cte = df['cte'].min(), df['cte'].max()
        min_theta_e, max_theta_e = df['theta_e'].min(), df['theta_e'].max()

        for d_idx in range(n_chunks):
            for v_idx in range(n_chunks):

                ax = axes[d_idx, v_idx]
                curr_df = df[(df['Chunk-d'] == d_idx + 1) & (df['Chunk-speed'] == v_idx + 1)]

                sns.scatterplot(ax=ax, data=curr_df, x='cte', y='theta_e', hue='Safe')

                cte_range = np.linspace(min_cte, max_cte, num=200)
                theta_e_range = np.linspace(min_theta_e, max_theta_e, num=200)
                avg_d = (self._bins_d[d_idx+1] + self._bins_d[d_idx]) / 2.
                avg_speed = (self._bins_v[v_idx+1] + self._bins_v[v_idx]) / 2.

                # ['cte', 'speed', 'theta_e', 'd']
                hvals = jax.vmap(
                    lambda s1: jax.vmap(
                        lambda s2: self._net.apply(params, jnp.array([s1, avg_speed, s2, avg_d]))
                    )(theta_e_range)
                )(cte_range).squeeze()

                cntr_plt = ax.contour(cte_range, theta_e_range, hvals, linewidths=3, colors='k')
                plt.clabel(cntr_plt, inline=1, fontsize=10)

                name = f'{self._bins_d[d_idx]:.2f} $\leq$ d $\leq$ {self._bins_d[d_idx+1]:.5f} and {self._bins_v[v_idx]:.5f} $\leq$ v $\leq$ {self._bins_v[v_idx+1]:.2f}'
                ax.set_title(name)
                ax.set_xlabel('cte')
                ax.set_ylabel('theta_e')

        wandb.log({'level_sets': wandb.Image(plt)})
        plt.savefig(os.path.join(self._results_path, 'level_sets.png'))
        plt.close()