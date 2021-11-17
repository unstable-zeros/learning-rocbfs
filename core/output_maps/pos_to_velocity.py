import jax.numpy as jnp
import numpy as np

class PosToVelocity:
    def __init__(self):
        pass

    @property
    def state_cols(self):
        return ['cte', 'v_est', 'theta_e', 'd']

    def map(self, df):

        df['x_diff'] = df['x-loc-center(m)'] - df['x-loc-center(m)'].shift(1)
        df['y_diff'] = df['y-loc-center(m)'] - df['y-loc-center(m)'].shift(1)
        df['tick_diff'] = df['Ticks(s)'] - df['Ticks(s)'].shift(1)
        
        # remove first row since it will have NaNs
        df.drop(df.head(1).index, inplace=True)

        def norm(*components):
            return np.sqrt(sum([c ** 2 for c in components]))

        df['pos_diff'] = df.apply(lambda x: norm(x['x_diff'], x['y_diff']), axis=1)
        df['v_est'] = df['pos_diff'] / df['tick_diff']

        return df

