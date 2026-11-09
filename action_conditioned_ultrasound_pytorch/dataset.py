import json
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .pose import relative_pose


class USVLAMetadata:
    def __init__(self, root, episode_indices=None):
        self.root = Path(root)
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        self.paths = sorted((self.root / 'data').glob('chunk-*/episode_*.parquet'))

        if episode_indices is not None:
            self.paths = [self.paths[i] for i in episode_indices]

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


class USVLAImageLoader:
    def __init__(
        self,
        metadata,
        image_size=(96, 128),
        image_mode='RGB',
        normalize='minus_one_one',
        center_crop=False,
        cache_size=2
    ):
        self.metadata = metadata
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.image_mode = image_mode
        self.normalize = normalize
        self.center_crop = center_crop
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def load_episode(self, episode_index):
        if episode_index not in self.cache:
            df = pd.read_parquet(
                self.metadata.paths[episode_index],
                columns=['frame_index', 'ultrasound_rgb']
            )

            self.cache[episode_index] = dict(zip(df['frame_index'], df['ultrasound_rgb']))

            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        self.cache.move_to_end(episode_index)

        return self.cache[episode_index]

    def __call__(self, episode_index, frame_index):
        data = self.load_episode(episode_index)[frame_index]

        if isinstance(data, dict):
            source = BytesIO(data['bytes']) if data['bytes'] is not None else self.metadata.root / data['path']
        else:
            source = BytesIO(data)

        with Image.open(source) as image:
            image = image.convert(self.image_mode)

            if self.center_crop:
                width, height = image.size
                side = min(width, height)
                left = (width - side) // 2
                top = (height - side) // 2
                image = image.crop((left, top, left + side, top + side))

            if self.image_size is not None:
                image = image.resize(self.image_size[::-1], Image.Resampling.BILINEAR)

            array = np.array(image, copy=True)

        if array.ndim == 2:
            array = array[..., None]

        image = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

        if self.normalize == 'minus_one_one':
            image = image * 2 - 1
        elif self.normalize != 'zero_one':
            raise ValueError(self.normalize)

        return image


class WorldModelDataset(Dataset):
    def __init__(
        self,
        metadata,
        image_loader=None,
        context_length=4,
        context_stride=1,
        future_offsets=(1, 5, 15),
        rotation_type='euler',
        image_size=(96, 128),
        image_mode='RGB',
        normalize='minus_one_one',
        center_crop=False,
        cache_size=2,
        relative_frame='tool'
    ):
        self.episodes = [metadata[i] for i in range(len(metadata))]
        self.image_loader = image_loader if image_loader is not None else USVLAImageLoader(
            metadata,
            image_size=image_size,
            image_mode=image_mode,
            normalize=normalize,
            center_crop=center_crop,
            cache_size=cache_size
        )
        self.context_length = context_length
        self.context_stride = context_stride
        self.rotation_type = rotation_type
        self.relative_frame = relative_frame

        self.indices = [
            (episode_index, t, offset)
            for episode_index, episode in enumerate(self.episodes)
            for t in range((context_length - 1) * context_stride, len(episode['frame_index']))
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
            torch.as_tensor(
                self.image_loader(episode_index, int(episode['frame_index'][i]))
            ).float()
            for i in range(
                t - (self.context_length - 1) * self.context_stride,
                t + 1,
                self.context_stride
            )
        ])

        target_image = torch.as_tensor(
            self.image_loader(episode_index, int(episode['frame_index'][target_t]))
        ).float()

        action = relative_pose(
            episode['state'][t],
            episode['state'][target_t],
            rotation_type=self.rotation_type,
            reference_frame=self.relative_frame
        )

        time_offset = episode['timestamp'][target_t] - episode['timestamp'][t]

        return {
            'context_images': context_images,
            'target_image': target_image,
            'action': torch.as_tensor(action, dtype=torch.float32),
            'time_offset': torch.as_tensor(time_offset, dtype=torch.float32)
        }