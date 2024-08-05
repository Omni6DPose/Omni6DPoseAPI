"""
3D/2D Affine Transformation
===========================
"""

import numpy as np
from numpy import ndarray

from .data_types import CameraIntrinsicsBase


def toAffine(scale_N: ndarray, rot_Nx3x3: ndarray, tr_Nx3: ndarray):
    """return affine transformation: [N, 4, 4]"""
    if scale_N.shape[0] == 0:
        return np.zeros((0, 4, 4)), np.zeros((0, 3))
    rot_Nx3x3 = rot_Nx3x3.reshape(-1, 3, 3) * scale_N.reshape(-1, 1, 1)
    tr_Nx3 = tr_Nx3.reshape(-1, 3)
    res = np.repeat(np.eye(4)[np.newaxis, ...], rot_Nx3x3.shape[0], axis=0)
    res[:, :3, :3] = rot_Nx3x3
    res[:, :3, 3] = tr_Nx3
    return res


def sRTdestruct(rts_Nx4x4: ndarray) -> "tuple[ndarray, ndarray, ndarray]":
    """Destruct multiple 4x4 affine transformation into scales, rotations and translations.

    :return: scales, rotations and translations each of shape [N], [N, 3, 3], [N, 3]
    """
    if rts_Nx4x4.shape[0] == 0:
        return np.zeros(0), np.zeros((0, 3, 3)), np.zeros((0, 3))
    scales = np.cbrt(np.linalg.det(rts_Nx4x4[:, :3, :3]))
    return (
        scales,
        (rts_Nx4x4[:, :3, :3] / scales.reshape(-1, 1, 1)),
        rts_Nx4x4[:, :3, 3],
    )


def depth2xyz(depth_img: ndarray, intrinsics: CameraIntrinsicsBase):
    """Transform ``depth_img`` pixel value to 3D coordinates under cv space, using camera intrinsics.

    About different camera space:

    - cv space: x goes right, y goes down, and you look through the positive direction of z axis
    - blender camera space: x goes right, y goes up, and you look through the negative direction of z axis

    :param depth_img: 2D matrix with shape (H, W)
    :returns: coordinates of each pixel with shape (H, W, 3)
    """

    # scale camera parameters
    h, w = depth_img.shape
    scale_x = w / intrinsics.width
    scale_y = h / intrinsics.height
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    x_offset = intrinsics.cx * scale_x
    y_offset = intrinsics.cy * scale_y

    indices = np.indices((h, w), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def pixel2xyz(h: int, w: int, pixel: ndarray, intrinsics: CameraIntrinsicsBase):
    """
    Transform `(pixel[0], pixel[1])` to normalized 3D vector under cv space, using camera intrinsics.

    :param h: height of the actual image
    :param w: width of the actual image
    """

    # scale camera parameters
    scale_x = w / intrinsics.width
    scale_y = h / intrinsics.height
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    x_offset = intrinsics.cx * scale_x
    y_offset = intrinsics.cy * scale_y

    x = (pixel[1] - x_offset) / fx
    y = (pixel[0] - y_offset) / fy
    vec = np.array([x, y, 1])
    return vec / np.linalg.norm(vec)


def transform_coordinates_3d(coordinates: ndarray, sRT: ndarray):
    """Apply 3D affine transformation to pointcloud.

    :param coordinates: ndarray of shape [3, N]
    :param sRT: ndarray of shape [4, 4]

    :returns: new pointcloud of shape [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack(
        [coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)]
    )
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d: ndarray, intrinsics_K: ndarray):
    """
    :param coordinates_3d: [3, N]
    :param intrinsics_K: K matrix [3, 3] (the return value of :meth:`.data_types.CameraIntrinsicsBase.to_matrix`)

    :returns: projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics_K @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates
