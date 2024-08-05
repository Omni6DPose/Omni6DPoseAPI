"""
Bounding Box IoU
================
"""

import numpy as np
from .box import Box as _Box
from .iou import IoU as _IoU


def compute_3d_bbox_iou(
    rA_3x3: np.ndarray,
    tA_3: np.ndarray,
    sA_3: np.ndarray,
    rB_3x3: np.ndarray,
    tB_3: np.ndarray,
    sB_3: np.ndarray,
) -> float:
    """Compute the exact 3D IoU of two boxes using `scipy.spatial.ConvexHull <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html>`_"""

    try:
        boxA = _Box.from_transformation(rA_3x3, tA_3, sA_3)
        boxB = _Box.from_transformation(rB_3x3, tB_3, sB_3)

        return _IoU(boxA, boxB).iou()
    except KeyboardInterrupt:
        raise
    except:  # silently ignore qhull error
        return 0
