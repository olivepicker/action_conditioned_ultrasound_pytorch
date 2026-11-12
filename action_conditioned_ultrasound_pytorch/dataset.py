import json
import os
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .pose import relative_pose


class SonATAMetadata:
    def __init__(self, root, split=None, episode_indices=None):
        self.root = Path(root)
        self.info = json.loads((self.root / 'meta/info.json').read_text())

        self.paths = sorted(
            (self.root / 'data').glob('chunk-*/episode_*.parquet'),
            key=lambda path: int(path.stem.split('_')[-1])
        )

        if split is not None:
            start, end = map(int, self.info['splits'][split].split(':'))
            self.paths = [
                path for path in self.paths
                if start <= int(path.stem.split('_')[-1]) < end
            ]

        if episode_indices is not None:
            indices = set(episode_indices)
            self.paths = [
                path for path in self.paths
                if int(path.stem.split('_')[-1]) in indices
            ]

        self.episode_ids = [
            int(path.stem.split('_')[-1])
            for path in self.paths
        ]

        with open(self.root / 'meta/episodes.jsonl') as f:
            self.lengths = {
                ep['episode_index']: ep['length']
                for ep in map(json.loads, f)
            }

    def __len__(self):
        return len(self.paths)

    def video_path(self, idx):
        episode_id = self.episode_ids[idx]
        chunk = episode_id // self.info['chunks_size']

        return (
            self.root
            / f'videos/chunk-{chunk:03d}'
            / 'observation.images.ultrasound'
            / f'episode_{episode_id:06d}.mp4'
        )

    def __getitem__(self, idx):
        df = pd.read_parquet(self.paths[idx], columns=[
            'episode_index',
            'frame_index',
            'timestamp',
            'action',
            'observation.state',
            'observation.meta.probe_cali_mtx',
            'task_index',
            'instruction.text',
            'instruction.task',
            'instruction.sub_task',
            'instruction.phantom'
        ])

        frame_index = df['frame_index'].to_numpy()

        if not np.array_equal(frame_index, np.arange(len(df))):
            raise ValueError('Non-contiguous frame indices')

        if len(df) != self.lengths[self.episode_ids[idx]]:
            raise ValueError('Episode length mismatch')

        return {
            'episode_index': df['episode_index'].to_numpy(),
            'frame_index': frame_index,
            'timestamp': df['timestamp'].to_numpy(),
            'command_pose': np.stack(df['action'].to_numpy()),
            'joint_state': np.stack(df['observation.state'].to_numpy()),
            'probe_calibration': np.stack(df['observation.meta.probe_cali_mtx'].to_numpy()),
            'task_index': df['task_index'].to_numpy(),
            'instruction': df['instruction.text'].tolist(),
            'task': df['instruction.task'].tolist(),
            'sub_task': df['instruction.sub_task'].tolist(),
            'phantom': df['instruction.phantom'].tolist()
        }


class SonATAImageLoader:
    def __init__(
        self,
        metadata,
        image_size=(96, 128),
        image_mode='RGB',
        normalize='minus_one_one',
        center_crop=False,
        cache_size=2,
        cache_dir=None
    ):
        self.metadata = metadata

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if image_size is None:
            height, width = metadata.info['features']['observation.images.ultrasound']['shape'][:2]

            if center_crop:
                height = width = min(height, width)

            image_size = (height, width)

        self.image_size = tuple(image_size)
        self.image_mode = image_mode
        self.normalize = normalize
        self.center_crop = center_crop
        self.cache_size = cache_size
        self.cache = OrderedDict()

        if image_mode not in ('RGB', 'L'):
            raise ValueError(image_mode)

        if normalize not in ('minus_one_one', 'zero_one'):
            raise ValueError(normalize)

        if cache_size < 1:
            raise ValueError('cache_size must be positive')

        self.cache_dir = None

        if cache_dir is not None:
            height, width = self.image_size
            name = f'{height}x{width}_{image_mode}_crop{int(center_crop)}'
            self.cache_dir = Path(cache_dir) / name
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def decode_episode(self, episode_index):
        height, width = self.image_size
        channels = 3 if self.image_mode == 'RGB' else 1
        pixel_format = 'rgb24' if channels == 3 else 'gray'

        filters = []

        if self.center_crop:
            source_shape = self.metadata.info['features']['observation.images.ultrasound']['shape']
            side = min(source_shape[:2])
            filters.append(f'crop={side}:{side}')

        filters.append(f'scale={width}:{height}:flags=bilinear')

        result = subprocess.run([
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-threads', '1',
            '-i', str(self.metadata.video_path(episode_index)),
            '-vf', ','.join(filters),
            '-pix_fmt', pixel_format,
            '-f', 'rawvideo',
            'pipe:1'
        ], capture_output=True, check=True)

        episode_id = self.metadata.episode_ids[episode_index]
        expected = self.metadata.lengths[episode_id]

        array = np.frombuffer(result.stdout, dtype=np.uint8)

        if array.size != expected * height * width * channels:
            raise RuntimeError(f'Video frame count mismatch: {episode_id}')

        return array.reshape(expected, height, width, channels).copy()

    def load_episode(self, episode_index):
        if episode_index not in self.cache:
            if self.cache_dir is None:
                frames = self.decode_episode(episode_index)
            else:
                episode_id = self.metadata.episode_ids[episode_index]
                path = self.cache_dir / f'episode_{episode_id:06d}.npy'

                if not path.exists():
                    frames = self.decode_episode(episode_index)
                    temp = path.with_name(
                        f'.{path.stem}.{os.getpid()}.npy'
                    )

                    np.save(temp, frames)
                    os.replace(temp, path)

                frames = np.load(path, mmap_mode='r')

            self.cache[episode_index] = frames

            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        self.cache.move_to_end(episode_index)

        return self.cache[episode_index]

    def __call__(self, episode_index, frame_index):
        frame = self.load_episode(episode_index)[frame_index]

        image = (
            torch.from_numpy(np.array(frame, copy=True))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        if self.normalize == 'minus_one_one':
            image = image * 2 - 1

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
        cache_dir=None,
        relative_frame='tool',
        output_rotation_type='euler'
    ):
        self.episodes = [metadata[i] for i in range(len(metadata))]

        self.image_loader = image_loader if image_loader is not None else SonATAImageLoader(
            metadata,
            image_size=image_size,
            image_mode=image_mode,
            normalize=normalize,
            center_crop=center_crop,
            cache_size=cache_size,
            cache_dir=cache_dir
        )

        self.context_length = context_length
        self.context_stride = context_stride
        self.rotation_type = rotation_type
        self.relative_frame = relative_frame
        self.output_rotation_type = output_rotation_type

        self.indices = [
            (episode_index, t, offset)
            for episode_index, episode in enumerate(self.episodes)
            for t in range(
                (context_length - 1) * context_stride,
                len(episode['frame_index'])
            )
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
            self.image_loader(
                episode_index,
                int(episode['frame_index'][i])
            )
            for i in range(
                t - (self.context_length - 1) * self.context_stride,
                t + 1,
                self.context_stride
            )
        ])

        target_image = self.image_loader(
            episode_index,
            int(episode['frame_index'][target_t])
        )

        action = relative_pose(
            episode['command_pose'][t],
            episode['command_pose'][target_t],
            rotation_type=self.rotation_type,
            reference_frame=self.relative_frame,
            output_rotation_type=self.output_rotation_type
        )

        time_offset = (
            episode['timestamp'][target_t]
            - episode['timestamp'][t]
        )

        return {
            'context_images': context_images,
            'target_image': target_image,
            'action': torch.as_tensor(action, dtype=torch.float32),
            'time_offset': torch.as_tensor(time_offset, dtype=torch.float32)
        }


class VAEDataset(Dataset):
    def __init__(
        self,
        metadata,
        image_loader=None,
        frame_stride=1,
        image_size=(96, 128),
        image_mode='RGB',
        normalize='minus_one_one',
        center_crop=False,
        cache_size=2,
        cache_dir=None
    ):
        if frame_stride < 1:
            raise ValueError('frame_stride must be positive')

        self.image_loader = image_loader if image_loader is not None else SonATAImageLoader(
            metadata,
            image_size=image_size,
            image_mode=image_mode,
            normalize=normalize,
            center_crop=center_crop,
            cache_size=cache_size,
            cache_dir=cache_dir
        )

        self.indices = [
            (episode_index, frame_index)
            for episode_index, episode_id in enumerate(metadata.episode_ids)
            for frame_index in range(
                0,
                metadata.lengths[episode_id],
                frame_stride
            )
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_index, frame_index = self.indices[idx]

        return self.image_loader(episode_index, frame_index)

    
class GoalPolicyDataset(Dataset):
    def __init__(
        self,
        world_dataset,
        action_offset=15,
        task_types=('navigation',)
    ):
        self.episodes = world_dataset.episodes
        self.image_loader = world_dataset.image_loader
        self.context_length = world_dataset.context_length
        self.context_stride = world_dataset.context_stride
        self.rotation_type = world_dataset.rotation_type
        self.relative_frame = world_dataset.relative_frame
        self.output_rotation_type = world_dataset.output_rotation_type
        self.action_offset = action_offset

        task_types = set(task_types) if task_types is not None else None

        self.indices = [
            (episode_index, t)
            for episode_index, episode in enumerate(self.episodes)
            if task_types is None or episode['task'][0] in task_types
            for t in range(
                (self.context_length - 1) * self.context_stride,
                len(episode['frame_index']) - action_offset
            )
        ]

    def __len__(self):
        return len(self.indices)

    def relative_action(self, source, target):
        return relative_pose(
            source,
            target,
            rotation_type=self.rotation_type,
            reference_frame=self.relative_frame,
            output_rotation_type=self.output_rotation_type
        )

    def __getitem__(self, idx):
        episode_index, t = self.indices[idx]
        episode = self.episodes[episode_index]

        context_indices = np.arange(
            t - (self.context_length - 1) * self.context_stride,
            t + 1,
            self.context_stride
        )

        target_t = t + self.action_offset
        goal_t = len(episode['frame_index']) - 1

        context_images = torch.stack([
            self.image_loader(
                episode_index,
                int(episode['frame_index'][i])
            )
            for i in context_indices
        ])

        goal_image = self.image_loader(
            episode_index,
            int(episode['frame_index'][goal_t])
        )

        poses = episode['command_pose']

        previous_actions = self.relative_action(
            poses[context_indices[:-1]],
            poses[context_indices[1:]]
        )

        expert_action = self.relative_action(
            poses[t],
            poses[target_t]
        )

        time_offset = (
            episode['timestamp'][target_t]
            - episode['timestamp'][t]
        )

        return {
            'context_images': context_images,
            'previous_actions': torch.as_tensor(
                previous_actions, dtype=torch.float32
            ),
            'goal_image': goal_image,
            'expert_action': torch.as_tensor(
                expert_action, dtype=torch.float32
            ),
            'time_offset': torch.as_tensor(
                time_offset, dtype=torch.float32
            ),
            'episode_id': int(episode['episode_index'][0])
        }