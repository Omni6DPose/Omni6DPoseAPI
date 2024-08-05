"""
Helper Functions
================

Common Helpers
--------------

.. autofunction:: mock_time

.. autofunction:: pool_map

.. autofunction:: save_pctxt

Visualization Helpers
---------------------

.. autofunction:: draw_bboxes

.. autofunction:: draw_text

.. autofunction:: draw_object_label

.. autofunction:: draw_3d_bbox

.. autofunction:: draw_pose_axes

"""

from contextlib import contextmanager
import logging
import time
from typing import Callable, TypeVar
from tqdm import tqdm
import multiprocessing.pool
import cv2
import numpy as np

from .bbox import create_3d_bbox
from .transform import (
    calculate_2d_projections,
    transform_coordinates_3d,
)
from .data_types import CameraIntrinsicsBase

# used for drawing
_bbox_coef = np.array(
    [
        [1, 1, 1],
        [1, 1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ]
)


def _draw_seg(
    img: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_color: np.ndarray,
    end_color: "np.ndarray | None" = None,
    thickness_coef=1,
):
    if end_color is None:
        end_color = start_color

    # automatically adjust line thickness according to image resolution
    thickness = round(img.shape[1] / 320 * thickness_coef)
    steps = 10
    step = (end - start) / steps
    s_step = (end_color - start_color) / steps
    for i in range(steps):
        x, y = start + (i * step), start + ((i + 1) * step)
        sx = start_color + ((i + 0.5) * s_step)
        x = tuple(x.astype(int))
        y = tuple(y.astype(int))
        sx = tuple(sx.astype(int))
        # set_trace()
        img = cv2.line(
            np.ascontiguousarray(img.copy()),
            x,
            y,
            (int(sx[0]), int(sx[1]), int(sx[2])),
            thickness,
        )

    return img


def draw_bboxes(img, projected_bbox_Nx2, transformed_bbox_Nx3, color=None, thickness=1):
    """
    :param color: can be a 3-element array/tuple.
    """
    if color is None:
        colors = (_bbox_coef * 255 + 255) / 2
    else:
        colors = np.tile(np.array(color).reshape(3), [8, 1])

    projected_bbox_Nx2 = np.int32(projected_bbox_Nx2).reshape(-1, 2)
    lines = [
        [4, 5],
        [5, 7],
        [6, 4],
        [7, 6],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [0, 1],
        [1, 3],
        [2, 0],
        [3, 2],
    ]
    # sort by distance
    lines.sort(
        reverse=True,
        key=lambda x: (
            (transformed_bbox_Nx3[x[0]] + transformed_bbox_Nx3[x[1]]) ** 2
        ).sum(),
    )
    for i, j in lines:
        img = _draw_seg(
            img,
            projected_bbox_Nx2[i],
            projected_bbox_Nx2[j],
            colors[i],
            colors[j],
            thickness_coef=thickness,
        )

    return img


def draw_text(img: np.ndarray, text: str, pos: np.ndarray) -> np.ndarray:
    """draw black text with red border"""
    text_args = {
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 0.4 * img.shape[1] / 640,
        "lineType": cv2.LINE_AA,
    }
    img = cv2.putText(
        img, str(text), org=pos, color=(255, 0, 0), thickness=6, **text_args
    )
    img = cv2.putText(
        img, str(text), org=pos, color=(0, 0, 0), thickness=1, **text_args
    )
    return img


def draw_3d_bbox(
    img: np.ndarray,
    intrinsics: CameraIntrinsicsBase,
    sRT_4x4: np.ndarray,
    bbox_side_len: "list[float]",
    color=None,
    thickness=1,
) -> np.ndarray:
    """Visualize predicted 3D bounding box.

    :param color: See :func:`draw_bboxes`.

    :return: The image with drawn bounding box.
    """

    # set_trace()
    matK = intrinsics.to_matrix()
    h, w = img.shape[0], img.shape[1]
    scale_y = h / intrinsics.height
    scale_x = w / intrinsics.width
    scale_2d = np.array([scale_x, scale_y])

    bbox_3d = create_3d_bbox(bbox_side_len)
    transformed_bbox_Nx3 = transform_coordinates_3d(bbox_3d, sRT_4x4)
    projected_bbox = calculate_2d_projections(transformed_bbox_Nx3, matK)
    projected_bbox = np.array(projected_bbox * scale_2d[None, ...], dtype=np.int32)
    img = draw_bboxes(
        img, projected_bbox, transformed_bbox_Nx3.T, color=color, thickness=thickness
    )

    return img


def draw_pose_axes(
    img: np.ndarray,
    intrinsics: CameraIntrinsicsBase,
    sRT_4x4: np.ndarray,
    length: float,
    thickness=1,
) -> np.ndarray:
    """Visualize predicted pose. The _XYZ_ axes are colored by _RGB_ respectively.

    :return: The image with drawn pose axes.
    """

    # set_trace()
    matK = intrinsics.to_matrix()
    h, w = img.shape[0], img.shape[1]
    scale_y = h / intrinsics.height
    scale_x = w / intrinsics.width
    scale_2d = np.array([scale_x, scale_y])

    axes_ends = transform_coordinates_3d(
        np.concatenate(
            [np.zeros((3, 1)), np.diag([length, length, length]) / 2], axis=1
        ),
        sRT_4x4,
    )
    origin, ax, ay, az = np.array(
        calculate_2d_projections(axes_ends, matK) * scale_2d[None, ...],
        dtype=np.int32,
    )
    img = _draw_seg(img, origin, ax, np.array([255, 0, 0]), thickness_coef=thickness)
    img = _draw_seg(img, origin, ay, np.array([0, 255, 0]), thickness_coef=thickness)
    img = _draw_seg(img, origin, az, np.array([0, 0, 255]), thickness_coef=thickness)

    return img


def draw_object_label(
    img: np.ndarray, intrinsics: CameraIntrinsicsBase, sRT_4x4: np.ndarray, label: str
) -> np.ndarray:
    """draw label text at the center of the object.

    :return: The image with drawn object label.
    """

    matK = intrinsics.to_matrix()
    h, w = img.shape[0], img.shape[1]
    scale_y = h / intrinsics.height
    scale_x = w / intrinsics.width
    scale_2d = np.array([scale_x, scale_y])

    pos3d = sRT_4x4[:3, 3].reshape(3, 1)
    pos = np.array(calculate_2d_projections(pos3d, matK)[0] * scale_2d, dtype=np.int32)

    img = draw_text(img, label, pos)

    return img


@contextmanager
def mock_time(
    fmt: str, logger: "logging.Logger | None" = None, use_print=False, debug=False
):
    """Record time usage.

    :param fmt: info print format
    :param use_print: use `print` instead of `logger.info` to display information
    :param logger: specify logger. Defaults to the ``'timer' logger of root``.
    :param debug: logging using debug level instead of info level.

    .. testsetup::

        from cutoop.utils import mock_time

    Usage:

    .. testcode::

        with mock_time("action took %.4fs"):
            # some heavy operation
            pass

    """
    if logger == None:
        logger = logging.root.getChild("timer")

    start = time.perf_counter()
    yield
    stop = time.perf_counter()

    if use_print:
        print(fmt % (stop - start))
    elif debug:
        logger.debug(fmt, stop - start)
    else:
        logger.info(fmt, stop - start)


def save_pctxt(
    file: str,
    pts: np.ndarray,
    rgb: "np.ndarray | None" = None,
    normal: "np.ndarray | None" = None,
):
    """Save point cloud array to file.

    :param pts: point cloud array with shape (n, 3)
    :param file: dest file name
    :param rgb: optional rgb data with shape (n, 3), **conflicting with normal**.
        The result format would be ``X Y Z R G B`` for one line.
    :param normal: optional normal data with shape (n, 3), **conflicting with rgb**.
        The result format would be ``X Y Z NX NY NZ`` for one line.
    """
    x, y, z = pts.T
    if rgb is not None:
        r, g, b = rgb.T
        with open(file, "w") as f:
            for i in range(x.shape[0]):
                f.write(f"{x[i]} {y[i]} {z[i]} {r[i]} {g[i]} {b[i]}\n")
    elif normal is not None:
        r, g, b = normal.T
        with open(file, "w") as f:
            for i in range(x.shape[0]):
                f.write(f"{x[i]} {y[i]} {z[i]} {r[i]} {g[i]} {b[i]}\n")
    else:
        np.savetxt(file, pts, fmt="%.6f")


_R = TypeVar("_R")


def pool_map(
    func: Callable[..., _R], items: list, processes=8, use_tqdm=True, desc="pool map"
) -> "list[_R]":
    """Thread pooling version of `map`. Equivalent to ``map(func, items)``, except that
    the argument should be a list (implementing ``__len__``) instead of an iterator.

    :param processes: number of threads.
    :param use_tqdm: whether to display a tqdm bar.
    :param desc: set the title of tqdm progress bar if ``use_tqdm`` is set.

    .. doctest::

        >>> def square(x):
        ...     return x * x
        >>> cutoop.utils.pool_map(square, list(range(10)))
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    """

    def run(pair):
        id, arg = pair
        return id, func(arg)

    with multiprocessing.pool.ThreadPool(processes=processes) as pool:
        mapper = pool.imap_unordered(run, enumerate(items))
        ret = [None] * len(items)
        if use_tqdm:
            mapper = tqdm(mapper, total=len(items), desc=f"{desc} (proc={processes})")
        for id, r in mapper:
            ret[id] = r
    return ret
