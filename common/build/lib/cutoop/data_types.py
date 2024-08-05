"""
Data Types
==========

Data types are defined to unify the parameter types of functions.
Moreover, they're able to be stored to and retrieved from JSON file.
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as _Rot


@dataclass
class Pose(object):
    """Object/Camera pose representation"""

    quaternion: "tuple[float, float, float, float]"
    """quaternion in scale-first format (wxyz)"""

    translation: "tuple[float, float, float]"
    """translation from object (centered) space to camera space"""

    def __post_init__(self):
        assert len(self.quaternion) == 4
        assert len(self.translation) == 3

    def to_affine(self, scale=None):
        """transform to affine transformation (with no additional scaling)

        :return: 4x4 numpy array.

        Here's an example of getting :class:`Pose` from rotation matrix:

        .. doctest::

            >>> from cutoop.data_types import Pose
            >>> from scipy.spatial.transform import Rotation
            >>> x, y, z, w = Rotation.from_matrix([
            ...     [ 0.,  0.,  1.],
            ...     [ 0.,  1.,  0.],
            ...     [-1.,  0.,  0.]
            ... ]).as_quat()
            >>> pose = Pose(quaternion=[w, x, y, z], translation=[1, 1, 1])
            >>> pose.to_affine()
            array([[ 0.,  0.,  1.,  1.],
                   [ 0.,  1.,  0.,  1.],
                   [-1.,  0.,  0.,  1.],
                   [ 0.,  0.,  0.,  1.]], dtype=float32)
        """
        q = self.quaternion
        rot = _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        if scale is not None:
            rot = rot * scale
        trans = np.array(self.translation)
        mtx = np.eye(4).astype(np.float32)
        mtx[:3, :3] = rot
        mtx[:3, 3] = trans
        return mtx


@dataclass
class CameraIntrinsicsBase:
    """Camera intrinsics data.
    The unit of ``fx, fy, cx, cy, width, height`` are all pixels.
    """

    fx: float  # unit: pixel
    fy: float  # unit: pixel
    cx: float  # unit: pixel
    cy: float  # unit: pixel
    width: float  # unit: pixel
    height: float  # unit: pixel

    def to_matrix(self):
        """Transform to 3x3 K matrix. i. e.::

            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1 ]]

        :return: 3x3 numpy array.
        """
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def fov_x(self):
        return np.rad2deg(2 * np.arctan2(self.width, 2 * self.fx))
