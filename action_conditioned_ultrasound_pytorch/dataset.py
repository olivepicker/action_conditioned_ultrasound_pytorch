import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from .pose import relative_pose


class USVLAMetadata:    # potentially unused
    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        self.paths = sorted((self.root / 'data').glob('chunk-*/episode_*.parquet'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        df = pd.read_parquet(self.paths[idx], columns=[
            'episode_index', 'frame_index', 'timestamp', 'state',
            'actions', 'task_index', 'prompt', 'us_gt', 'stage'
        ])

        return {
            'episode_index': df['episode_index'].to_numpy(),
            'frame_index': df['frame_index'].to_numpy(),
            'timestamp': df['timestamp'].to_numpy(),
            'state': np.stack(df['state'].to_numpy()),
            'actions': np.stack(df['actions'].to_numpy()),
            'task_index': df['task_index'].to_numpy(),
            'prompt': df['prompt'].tolist(),
            'us_gt': df['us_gt'].to_numpy(),
            'stage': df['stage'].to_numpy()
        }


class WorldModelDataset(Dataset):
    def __init__(self, metadata, image_loader, context_length=4, future_offsets=(1, 5, 15), rotation_type='euler'):
        self.episodes = [metadata[i] for i in range(len(metadata))]
        self.image_loader = image_loader
        self.context_length = context_length
        self.rotation_type = rotation_type
        self.indices = [
            (episode_index, t, offset)
            for episode_index, episode in enumerate(self.episodes)
            for t in range(context_length - 1, len(episode['frame_index']))
            for offset in future_offsets
            if t + offset < len(episode['frame_index'])
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_index, t, offset = self.indices[idx]
        episode = self.episodes[episode_index]
        target_t = t + offset

        context_images = torch.stack([
            torch.as_tensor(self.image_loader(episode_index, int(episode['frame_index'][i]))).float()
            for i in range(t - self.context_length + 1, t + 1)
        ])
        target_image = torch.as_tensor(
            self.image_loader(episode_index, int(episode['frame_index'][target_t]))
        ).float()

        action = relative_pose(
            episode['state'][t], episode['state'][target_t], rotation_type=self.rotation_type
        )
        time_offset = episode['timestamp'][target_t] - episode['timestamp'][t]

        return {
            'context_images': context_images,
            'target_image': target_image,
            'action': torch.as_tensor(action, dtype=torch.float32),
            'time_offset': torch.as_tensor(time_offset, dtype=torch.float32)
        }
