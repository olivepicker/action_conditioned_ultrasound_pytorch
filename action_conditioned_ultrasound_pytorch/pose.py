import numpy as np
from scipy.spatial.transform import Rotation


def relative_pose(source, target, rotation_type='euler'):
    source = np.asarray(source)[..., :6]
    target = np.asarray(target)[..., :6]

    if rotation_type == 'euler':
        rotation_source = Rotation.from_euler('xyz', source[..., 3:6])
        rotation_target = Rotation.from_euler('xyz', target[..., 3:6])
    elif rotation_type == 'rotvec':
        rotation_source = Rotation.from_rotvec(source[..., 3:6])
        rotation_target = Rotation.from_rotvec(target[..., 3:6])
    else:
        raise ValueError(rotation_type)

    rotation = rotation_source.inv() * rotation_target
    translation = rotation_source.inv().apply(target[..., :3] - source[..., :3])

    return np.concatenate((translation, rotation.as_rotvec()), axis=-1)
