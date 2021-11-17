import numpy as np
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset
import torch
import os
import pandas as pd

DATA_ROOT = 'data/carla/images/Dataset_with_image/left_turn_state_space_sampling'
DATA_DIRS = [
    'random_noise_driving',
    'straight_lane_driving',
    'start_of_turn',
    'middle_of_turn'
]
FNAME = 'Data_Collection_Compiled.pd'
MEAN_IMG = [142.30428901, 142.93909777, 125.39351831]
STD_IMG = [58.4192335,  53.82467569, 61.2449872]

def loader(train_bs, test_bs):
    train_data = CarlaDataset(labels=['cte'], train=True)
    train_loader = NumpyLoader(train_data, batch_size=train_bs)

    test_data = CarlaDataset(labels=['cte'], train=False)
    test_loader = NumpyLoader(test_data, batch_size=test_bs)

    return train_loader, test_loader

class CarlaDataset(Dataset):
    def __init__(self, labels, train=True):

        data = self._read()

        # if train is True:
        #     data = data.head(n=int(0.8 * len(data)))
        # else:
        #     data = data.tail(n=int(0.2 * len(data)))

        # data = data.head(1000)

        self.images = np.stack(data['front_camera_image'].to_numpy(), axis=0)
        self.states = data[['speed(m/s)', 'theta_e', 'd']]
        self.labels = data[labels].to_numpy().reshape(len(data),)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self._normalize(self.images[index]), self.labels[index]

    @staticmethod
    def _read():
        all_dfs = []
        for dir_name in DATA_DIRS:
            path = os.path.join(DATA_ROOT, dir_name, FNAME)
            df = pd.read_pickle(path)
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)

    @staticmethod
    def _normalize(image):
        image = image.astype(np.float32)
        image -= np.array(MEAN_IMG, dtype=np.float32).reshape(1, 1, 3)
        image /= np.array(STD_IMG, dtype=np.float32).reshape(1, 1, 3)
        return image


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def jnp_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(jnp_collate(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)

class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=jnp_collate,
        pin_memory=pin_memory,
        drop_last=drop_last)
