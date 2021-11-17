import jax.numpy as jnp
import numpy as np
import os
import pickle
from haiku.nets import ResNet18
import haiku as hk
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from functools import partial
import jax


from core.data.images import NumpyLoader


class ImgToCTE:
    def __init__(self):

        self.forward = hk.transform_with_state(self._forward)
        self.params, self.nn_state = self.load_checkpoint()

    @property
    def state_cols(self):
        return ['cte_est', 'speed(m/s)', 'theta_e', 'd']

    def map(self, df):

        image_dataset = OutputMapDataset(df)
        loader = NumpyLoader(image_dataset, batch_size=64)

        df['cte_est'] = np.squeeze(np.vstack(
            [self.predict(imgs) for imgs in tqdm(loader)]))
        df = df.drop(columns=['front_camera_image'])
        return df

    @staticmethod
    def _forward(images, is_training):
        return ResNet18(1, resnet_v2=True)(images, is_training)

    @partial(jax.jit, static_argnums=0)
    def predict(self, imgs):
        pred, _ = self.forward.apply(
            self.params,
            self.nn_state,
            None,
            imgs,
            is_training=False)
        return jnp.asarray(pred)

    @staticmethod
    def load_checkpoint():
        dirname = '../old_trained_results/0904_output_map/perception-ckpts'
        with open(os.path.join(dirname, 'params.npy'), 'rb') as f:
            params = pickle.load(f)
        with open(os.path.join(dirname, 'nn_state.npy'), 'rb') as f:
            nn_state = pickle.load(f)

        return params, nn_state


class OutputMapDataset(Dataset):

    MEAN_IMG = [142.30428901, 142.93909777, 125.39351831]
    STD_IMG = [58.4192335, 53.82467569, 61.2449872]

    def __init__(self, df):
        self.images = np.stack(df['front_camera_image'].to_numpy(), axis=0)

    def __getitem__(self, index):
        return self._normalize(self.images[index])

    def __len__(self):
        return self.images.shape[0]

    def _normalize(self, image):
        image = image.astype(np.float32)
        image -= np.array(self.MEAN_IMG, dtype=np.float32).reshape(1, 1, 3)
        image /= np.array(self.STD_IMG, dtype=np.float32).reshape(1, 1, 3)
        return image

    @staticmethod
    def normalize(image):
        MEAN_IMG = [142.30428901, 142.93909777, 125.39351831]
        STD_IMG = [58.4192335, 53.82467569, 61.2449872]
        image = image.astype(np.float32)
        image -= np.array(MEAN_IMG, dtype=np.float32).reshape(1, 1, 3)
        image /= np.array(STD_IMG, dtype=np.float32).reshape(1, 1, 3)
        return image
