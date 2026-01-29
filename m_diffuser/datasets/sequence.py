import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryNormalizer:
    """
    Normalizes concatenated trajectories [obs, act] along the last dimension.
    Observations are normalized; actions are left as-is.
    """
    def __init__(self, obs_mean, obs_std, act_dim: int):
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        self.act_dim = int(act_dim)


    def normalize_observations(self, obs):
        # obs may be dict or array/tensor
        if isinstance(obs, dict):
            obs = _flatten_obs(obs)

        if isinstance(obs, torch.Tensor):
            x = obs.detach().cpu().numpy()
            x = np.asarray(x, dtype=np.float32)
            d = x.shape[-1]
            mean = self.obs_mean[:d]
            std = self.obs_std[:d]
            y = ((x - mean) / (std + 1e-6)).astype(np.float32)
            return torch.from_numpy(y).to(obs.device)

        x = np.asarray(obs, dtype=np.float32)
        d = x.shape[-1]
        mean = self.obs_mean[:d]
        std = self.obs_std[:d]
        return ((x - mean) / (std + 1e-6)).astype(np.float32)

    def unnormalize_observations(self, obs):
        if isinstance(obs, dict):
            obs = _flatten_obs(obs)

        if isinstance(obs, torch.Tensor):
            x = obs.detach().cpu().numpy()
            x = np.asarray(x, dtype=np.float32)
            d = x.shape[-1]
            mean = self.obs_mean[:d]
            std = self.obs_std[:d]
            y = (x * (std + 1e-6) + mean).astype(np.float32)
            return torch.from_numpy(y).to(obs.device)

        x = np.asarray(obs, dtype=np.float32)
        d = x.shape[-1]
        mean = self.obs_mean[:d]
        std = self.obs_std[:d]
        return (x * (std + 1e-6) + mean).astype(np.float32)

    def normalize_actions(self, act):
        # actions were not normalized in our dataset pipeline
        return act

    def unnormalize_actions(self, act):
        return act

    def _split(self, x):
        obs = x[..., :-self.act_dim]
        act = x[..., -self.act_dim:]
        return obs, act

    def normalize(self, traj):
        if isinstance(traj, torch.Tensor):
            x = traj.detach().cpu().numpy()
            obs, act = self._split(x)
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-6)
            y = np.concatenate([obs, act], axis=-1).astype(np.float32)
            return torch.from_numpy(y).to(traj.device)
        x = np.asarray(traj)
        obs, act = self._split(x)
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        return np.concatenate([obs, act], axis=-1).astype(np.float32)

    def unnormalize(self, traj):
        if isinstance(traj, torch.Tensor):
            x = traj.detach().cpu().numpy()
            obs, act = self._split(x)
            obs = obs * (self.obs_std + 1e-6) + self.obs_mean
            y = np.concatenate([obs, act], axis=-1).astype(np.float32)
            return torch.from_numpy(y).to(traj.device)
        x = np.asarray(traj)
        obs, act = self._split(x)
        obs = obs * (self.obs_std + 1e-6) + self.obs_mean
        return np.concatenate([obs, act], axis=-1).astype(np.float32)


def _flatten_obs(obs):
    """
    Minari PointMaze observations are often dicts:
      {'observation', 'desired_goal', 'achieved_goal'}
    Flatten into a single vector per timestep.
    """
    if isinstance(obs, dict):
        parts = []
        for k in ["observation", "desired_goal", "achieved_goal"]:
            if k in obs:
                parts.append(np.asarray(obs[k]))
        if not parts:
            for k in sorted(obs.keys()):
                parts.append(np.asarray(obs[k]))
        return np.concatenate(parts, axis=-1)
    return np.asarray(obs)


class SequenceDataset(Dataset):
    """
    Loads a Minari dataset (e.g. D4RL/pointmaze/umaze-v2) and windows it into
    fixed-length sequences of length `horizon`.

    __getitem__ returns:
      batch['conditions'] : [H, obs_dim + act_dim]
    and also observations/actions for convenience.
    """

    def __init__(
        self,
        dataset_name: str = None,
        env_name: str = None,
        horizon: int = 32,
        normalize_obs: bool = True,
        download: bool = True,
        **kwargs,
    ):
        if env_name is None:
            env_name = dataset_name
        if env_name is None:
            raise ValueError("Need dataset_name or env_name")

        self.dataset_name = str(env_name)
        self.horizon = int(horizon)
        self.normalize_obs = bool(normalize_obs)

        # --- load episodes via Minari ---
        import minari
        ds = minari.load_dataset(self.dataset_name, download=download)

        self.episodes = []
        for ep in ds.iterate_episodes():
            obs = _flatten_obs(ep.observations)   # length T+1
            acts = np.asarray(ep.actions)         # length T

            obs = obs[:-1]  # align obs to actions length

            T = min(obs.shape[0], acts.shape[0])
            if T < self.horizon:
                continue

            obs = obs[:T].astype(np.float32)
            acts = acts[:T].astype(np.float32)
            self.episodes.append((obs, acts))

        if not self.episodes:
            raise ValueError(f"No episodes with horizon={self.horizon} found in {self.dataset_name}")

        # --- build indices ---
        self.indices = []
        for epi, (obs, _acts) in enumerate(self.episodes):
            T = obs.shape[0]
            for t in range(0, T - self.horizon + 1):
                self.indices.append((epi, t))

        # --- dims expected by train/eval scripts ---
        self.obs_dim = self.episodes[0][0].shape[-1]
        self.act_dim = self.episodes[0][1].shape[-1]
        self.observation_dim = self.obs_dim
        self.action_dim = self.act_dim
        self.transition_dim = self.observation_dim + self.action_dim

        # --- normalization stats (always defined) ---
        if self.normalize_obs:
            all_obs = np.concatenate([ep[0] for ep in self.episodes], axis=0)
            self.obs_mean = all_obs.mean(axis=0).astype(np.float32)
            self.obs_std = (all_obs.std(axis=0) + 1e-6).astype(np.float32)
        else:
            self.obs_mean = np.zeros(self.observation_dim, dtype=np.float32)
            self.obs_std = np.ones(self.observation_dim, dtype=np.float32)

        # normalizer used by evaluate.py
        self.normalizer = TrajectoryNormalizer(self.obs_mean, self.obs_std, self.action_dim)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        epi, t = self.indices[idx]
        obs, acts = self.episodes[epi]

        o = obs[t:t + self.horizon]
        a = acts[t:t + self.horizon]

        if self.normalize_obs:
            o = (o - self.obs_mean) / (self.obs_std + 1e-6)

        traj = np.concatenate([o, a], axis=-1).astype(np.float32)

        return {
            "conditions": torch.from_numpy(traj),                 # [H, obs_dim + act_dim]
            "observations": torch.from_numpy(o.astype(np.float32)),
            "actions": torch.from_numpy(a.astype(np.float32)),
        }


def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 4, shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=True,
    )
