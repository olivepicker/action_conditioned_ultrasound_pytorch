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


def fit_paper_action_bins(
    world_dataset,
    action_offset=15,
    task_types=('navigation',),
    scale_quantile=0.99,
    bin_quantiles=(0.20, 0.65)
):
    actions = []

    for episode in world_dataset.episodes:
        if episode['task'][0] not in task_types:
            continue

        poses = episode['command_pose']

        if len(poses) <= action_offset:
            continue

        relative_actions = relative_pose(
            poses[:-action_offset],
            poses[action_offset:],
            rotation_type='euler',
            reference_frame='tool',
            output_rotation_type='euler'
        )

        actions.append(relative_actions)

    if not actions:
        raise ValueError('No eligible training actions found')

    actions = np.concatenate(actions, axis=0)

    min_scales = np.array([
        1e-5, 1e-5, 1e-5,
        1e-3, 1e-3, 1e-3
    ])

    scales = np.maximum(
        np.quantile(np.abs(actions), scale_quantile, axis=0),
        min_scales
    )

    normalized = actions / scales
    dominant_axes = np.abs(normalized).argmax(axis=1)

    dominant_values = normalized[
        np.arange(len(actions)),
        dominant_axes
    ]

    thresholds = np.quantile(
        np.abs(dominant_values),
        bin_quantiles
    )

    if not 0 < thresholds[0] < thresholds[1]:
        raise ValueError('Invalid bin thresholds')

    return scales.astype(np.float32), thresholds.astype(np.float32)


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
        action_scales,
        bin_thresholds,
        goal_bank,
        action_offset=15,
        horizon=4,
        task_types=('navigation',)
    ):
        self.world_dataset = world_dataset
        self.episodes = world_dataset.episodes
        self.image_loader = world_dataset.image_loader

        self.context_length = world_dataset.context_length
        self.context_stride = world_dataset.context_stride

        self.action_offset = action_offset
        self.horizon = horizon
        self.goal_bank = goal_bank

        self.action_scales = np.asarray(action_scales, dtype=np.float32)
        self.bin_thresholds = np.asarray(bin_thresholds, dtype=np.float32)

        if self.action_scales.shape != (6,):
            raise ValueError('action_scales must have shape (6,)')

        if self.bin_thresholds.shape != (2,):
            raise ValueError('bin_thresholds must have shape (2,)')

        if not 0 < self.bin_thresholds[0] < self.bin_thresholds[1]:
            raise ValueError('Invalid bin thresholds')

        self.indices = []
        self.episode_actions = {}

        context_start = (self.context_length - 1) * self.context_stride

        for e, episode in enumerate(self.episodes):
            if episode['task'][0] not in task_types:
                continue

            poses = episode['command_pose']
            length = len(poses)

            if length <= context_start + horizon * action_offset:
                continue

            self.episode_actions[e] = relative_pose(
                poses[:-action_offset],
                poses[action_offset:],
                rotation_type='euler',
                reference_frame='tool',
                output_rotation_type='euler'
            ).astype(np.float32)

            for t in range(context_start, length - horizon * action_offset):
                self.indices.append((e, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        e, t = self.indices[index]
        episode = self.episodes[e]

        context_indices = (
            t - np.arange(self.context_length - 1, -1, -1)
            * self.context_stride
        )

        frames = episode['frame_index']

        context_images = torch.stack([
            self.image_loader(e, int(frames[i]))
            for i in context_indices
        ])

        episode_id = int(episode['episode_index'][0])

        goal_images = self.goal_bank(
            sub_task=str(episode['sub_task'][0]),
            sample_index=index,
            exclude_episode_id=episode_id
        )

        poses = episode['command_pose']

        previous_actions = relative_pose(
            poses[context_indices[:-1]],
            poses[context_indices[1:]],
            rotation_type='euler',
            reference_frame='tool',
            output_rotation_type='euler'
        ).astype(np.float32)

        starts = t + np.arange(self.horizon) * self.action_offset
        ends = starts + self.action_offset

        actions = self.episode_actions[e][starts].copy()

        normalized = actions / self.action_scales
        axis_targets = np.abs(normalized).argmax(axis=1)

        selected_values = normalized[
            np.arange(self.horizon),
            axis_targets
        ]

        small, large = self.bin_thresholds

        bin_targets = np.digitize(
            selected_values,
            bins=[-large, -small, small, large]
        ).astype(np.int64)

        axis_mask = bin_targets != 2

        timestamps = episode['timestamp']

        time_offsets = (
            timestamps[ends] - timestamps[t]
        ).astype(np.float32)

        return {
            'context_images': context_images,
            'previous_actions': torch.from_numpy(previous_actions),
            'goal_images': goal_images,
            'expert_action': torch.from_numpy(actions[0].copy()),
            'action_sequence': torch.from_numpy(actions),
            'axis_targets': torch.from_numpy(axis_targets.astype(np.int64)),
            'bin_targets': torch.from_numpy(bin_targets),
            'axis_mask': torch.from_numpy(axis_mask),
            'stop_targets': torch.zeros(self.horizon),
            'stop_mask': torch.zeros(self.horizon, dtype=torch.bool),
            'time_offset': torch.tensor(time_offsets[0]),
            'time_offsets': torch.from_numpy(time_offsets),
            'episode_id': int(episode['episode_index'][0]),
            'sub_task': episode['sub_task'][0]
        }


class GoalBank:
    def __init__(
        self,
        metadata,
        image_loader,
        num_goals=4,
        task_types=('navigation',),
        seed=42
    ):
        self.metadata = metadata
        self.image_loader = image_loader
        self.num_goals = num_goals
        self.seed = seed
        self.groups = {}

        for e in range(len(metadata)):
            episode = metadata[e]

            if episode['task'][0] not in task_types:
                continue

            sub_task = str(episode['sub_task'][0])
            episode_id = int(episode['episode_index'][0])
            frame_index = int(episode['frame_index'][-1])

            self.groups.setdefault(sub_task, []).append(
                (e, episode_id, frame_index)
            )

    def __call__(
        self,
        sub_task,
        sample_index=0,
        exclude_episode_id=None
    ):
        candidates = [
            item
            for item in self.groups[sub_task]
            if item[1] != exclude_episode_id
        ]

        if not candidates:
            raise ValueError(f'No goal candidates for {sub_task}')

        rng = np.random.default_rng(self.seed + int(sample_index))

        selected = rng.choice(
            len(candidates),
            size=self.num_goals,
            replace=len(candidates) < self.num_goals
        )

        images = [
            self.image_loader(
                candidates[int(i)][0],
                candidates[int(i)][2]
            )
            for i in selected
        ]

        return torch.stack(images)