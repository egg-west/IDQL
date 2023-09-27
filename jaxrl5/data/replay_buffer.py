from typing import Optional, Union, Dict, Iterable, Tuple
import os
import pickle

import h5py
import tqdm
import gym
import gym.spaces
import numpy as np

from jaxrl5.data.dataset import Dataset, DatasetDict

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()

def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        else:
            raise TypeError("Unsupported type.")
        new_dataset_dict[k] = new_v
    return new_dataset_dict

def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len

class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def normalize_state(self, eps=1e-3):
        mean = self.dataset_dict["observations"].mean(0, keepdims=True)
        std = self.dataset_dict["observations"].std(0, keepdims=True) + eps

        self.dataset_dict["observations"] = (self.dataset_dict["observations"] - mean) / std
        self.dataset_dict["next_observations"] = (self.dataset_dict["next_observations"] - mean) / std

        return mean, std

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        for i in range(len(self)):
            episode_return += self.dataset_dict["rewards"][i]

            if self.dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns
    
    def filter(
        self, percentile: Optional[float] = None, threshold: Optional[float] = None
    ):
        assert (percentile is None and threshold is not None) or (
            percentile is not None and threshold is None
        )

        (
            episode_starts,
            episode_ends,
            episode_returns,
        ) = self._trajectory_boundaries_and_returns()

        if percentile is not None:
            threshold = np.percentile(episode_returns, 100 - percentile)

        bool_indx = np.full((len(self),), False, dtype=bool)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i] : episode_ends[i]] = True

        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)

        self.dataset_len = _check_lengths(self.dataset_dict)
        self._size = self.dataset_dict["dones"].shape[0]

    def normalize_returns(self, scaling: float = 1000):
        (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
        self.dataset_dict["rewards"] /= np.max(episode_returns) - np.min(
            episode_returns
        )
        self.dataset_dict["rewards"] *= scaling
    
    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        assert 0 < ratio and ratio < 1
        train_index = np.index_exp[: int(self.dataset_len * ratio)]
        test_index = np.index_exp[int(self.dataset_len * ratio) :]

        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio) :]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def save(self, path: str, file_name: str):
        assert os.path.exists(path)
        with open(os.path.join(path, file_name), "wb") as f:
            pickle.dump([self.dataset_dict, self._size], f)

    def load(self, path: str, file_name: str):
        if 'pkl' in file_name:
            self.load_pkl(path, file_name)
        elif 'hdf5' in file_name:
            self.load_hdf5(path, file_name)
        else:
            raise NotImplementedError

    def load_pkl(self, path: str, file_name: str):
        f_path = os.path.join(path, file_name)
        assert os.path.exists(f_path)

        with open(f_path, "rb") as f:
            meta_data = pickle.load(f)
            assert len(meta_data) == 2
            self._size = meta_data[1]
            self.dataset_dict = {}
            for k, v in meta_data[0].items():
                #print(type(v))
                if (type(v) == type(np.zeros((2, 2)))):
                    self.dataset_dict[k] = v[:self._size, :]
                #    print(k, v.shape)
            #self.dataset_dict = meta_data[0]
        # TODO change the data collection to delete this line
        self.dataset_dict['rewards'] = np.squeeze(self.dataset_dict['rewards'], 1)
        self.dataset_dict['dones'] = np.squeeze(self.dataset_dict['terminals'], 1)
        self.dataset_dict['masks'] = -self.dataset_dict['dones'].astype(np.float32) + 1.0
    
    def load_hdf5(self, path: str, file_name: str):
        f_path = os.path.join(path, file_name)
        assert os.path.exists(f_path)
        assert "hdf5" in file_name
        
        with h5py.File(f_path, 'r') as dataset_file:
            #for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            for k in ['observations', 'next_observations', 'actions', 'rewards', ]:
                try:  # first try loading as an array
                    self.dataset_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    self.dataset_dict[k] = dataset_file[k][()]
            self.dataset_dict['dones'] = dataset_file['terminals'][:]
            self.dataset_dict['masks'] = -self.dataset_dict['dones'].astype(np.float32) + 1.0
            self._size = dataset_file['terminals'].shape[0]
