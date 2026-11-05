import json
from pathlib import Path

import numpy as np
import pandas as pd


class USVLAMetadata:
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
