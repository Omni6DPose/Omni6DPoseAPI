"""
Bounding Box Utilities
======================
"""

import numpy as np

CUBE_3x8 = np.array(
    [
        [+1, +1, +1],
        [+1, +1, -1],
        [-1, +1, +1],
        [-1, +1, -1],
        [+1, -1, +1],
        [+1, -1, -1],
        [-1, -1, +1],
        [-1, -1, -1],
    ]
).T
"""cube from (-1, -1, -1) to (1, 1, 1).

The index assignment is::

    ...   6------4
    ...  /|     /|
    ... 2-+----0 |
    ... | 7----+-5
    ... |/     |/
    ... 3------1

"""


def create_3d_bbox(size: "list[float] | np.ndarray | float"):
    """Create a 3d aabb bbox centered at origin with specific size.

    :param size: bound box side length.

    :returns: ndarray of shape [3, N].

    The index assignment is::

        ...   6------4
        ...  /|     /|
        ... 2-+----0 |
        ... | 7----+-5
        ... |/     |/
        ... 3------1

    """
    if not hasattr(size, "__iter__"):
        size = [size, size, size]
    assert len(size) == 3
    bbox_3d = CUBE_3x8 * np.array(size).reshape(3, 1) / 2
    return bbox_3d


def create_3d_bbox_pc(size: "list[float] | np.ndarray | float"):
    """Generate point cloud for a bounding box. Mainly used for debuging.

    :return: numpy array of shape (3, ?)
    """
    if not hasattr(size, "__iter__"):
        size = [size, size, size]
    assert len(size) == 3

    xs = np.linspace(-size[0] / 2, size[0] / 2, 40).reshape(-1, 1) * np.array([1, 0, 0])
    ys = np.linspace(-size[1] / 2, size[1] / 2, 40).reshape(-1, 1) * np.array([0, 1, 0])
    zs = np.linspace(-size[2] / 2, size[2] / 2, 40).reshape(-1, 1) * np.array([0, 0, 1])
    A = (
        xs.reshape(-1, 1, 1, 3)
        + ys.reshape(1, -1, 1, 3)
        + zs[[0, -1]].reshape(1, 1, -1, 3)
    )
    B = (
        xs.reshape(-1, 1, 1, 3)
        + ys[[0, -1]].reshape(1, -1, 1, 3)
        + zs.reshape(1, 1, -1, 3)
    )
    C = (
        xs[[0, -1]].reshape(-1, 1, 1, 3)
        + ys.reshape(1, -1, 1, 3)
        + zs.reshape(1, 1, -1, 3)
    )
    box_pc = np.concatenate([A.reshape(-1, 3), B.reshape(-1, 3), C.reshape(-1, 3)])
    return box_pc.T
