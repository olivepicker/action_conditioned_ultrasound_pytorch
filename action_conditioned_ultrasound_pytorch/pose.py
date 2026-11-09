import numpy as np
from scipy.spatial.transform import Rotation


def relative_pose(source, target, rotation_type='euler', reference_frame='tool'):
    source = np.asarray(source, dtype=np.float64)[..., :6]
    target = np.asarray(target, dtype=np.float64)[..., :6]

    if source.shape != target.shape or source.shape[-1] != 6:
        raise ValueError('source and target must have matching pose shapes')

    if rotation_type == 'euler':
        rotation_source = Rotation.from_euler('xyz', source[..., 3:6])
        rotation_target = Rotation.from_euler('xyz', target[..., 3:6])
    elif rotation_type == 'rotvec':
        rotation_source = Rotation.from_rotvec(source[..., 3:6])
        rotation_target = Rotation.from_rotvec(target[..., 3:6])
    else:
        raise ValueError(rotation_type)

    translation = target[..., :3] - source[..., :3]

    if reference_frame == 'tool':
        rotation = rotation_source.inv() * rotation_target
        translation = rotation_source.inv().apply(translation)
    elif reference_frame == 'base':
        rotation = rotation_target * rotation_source.inv()
    else:
        raise ValueError(reference_frame)

    return np.concatenate((translation, rotation.as_rotvec()), axis=-1)