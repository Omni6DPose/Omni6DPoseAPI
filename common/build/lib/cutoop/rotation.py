"""
Rotation and Symmetry
=====================

You can add ``CUTOOP_STRICT_MODE=1`` to environment to enable a strict value assertion
before calculating rotation error.

SymLabel
--------

.. autoclass:: SymLabel
   :show-inheritance:
   :members:
   :special-members: __str__

Rotation Manipulation
---------------------

.. autofunction:: rot_canonical_sym

.. autofunction:: rots_of_sym

Rotation Error Computation
--------------------------

.. autofunction:: rot_diff_axis

.. autofunction:: rot_diff_sym

.. autofunction:: rot_diff_theta

.. autofunction:: rot_diff_theta_pointwise

Prelude Rotations
-----------------

.. autodata:: STANDARD_SYMMETRY

.. autodata:: cube_group

.. autodata:: cube_flip_group

.. autodata:: qtr

.. autodata:: ax

.. autodata:: rot90

.. autodata:: rot180

"""

try:  # just for type hints
    from typing import Literal
except:
    pass

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation as _Rotation
from dataclasses import dataclass
import copy
import os


@dataclass
class SymLabel:
    """
    Symmetry labels for real-world objects.

    Axis rotation details:

    - `any`: arbitrary rotation around this axis is ok
    - `half`: rotate 180 degrees along this axis (central symmetry)
    - `quarter`: rotate 90 degrees along this axis (like square)

    .. doctest::

        >>> from cutoop.rotation import SymLabel
        >>> sym = SymLabel(any=False, x='any', y='none', z='none')
        >>> str(sym)
        'x-cone'
        >>> sym = SymLabel(any=False, x='any', y='any', z='none') # two any produces 'any'
        >>> str(sym)
        'any'
        >>> sym = SymLabel(any=False, x='half', y='half', z='half')
        >>> str(sym)
        'box'

    """

    any: bool  # sphere
    """Whether arbitrary rotation is allowed"""
    x: "Literal['none', 'any', 'half', 'quarter']"
    """axis rotation for x"""
    y: "Literal['none', 'any', 'half', 'quarter']"
    """axis rotation for y"""
    z: "Literal['none', 'any', 'half', 'quarter']"
    """axis rotation for z"""

    def get_only(self, tag: 'Literal["any", "half", "quarter"]'):
        """Get the only axis marked with the tag. If multiple or none is find, return ``None``."""
        ret: 'list[Literal["x", "y", "z"]]' = []
        if self.x == tag:
            ret.append("x")
        if self.y == tag:
            ret.append("y")
        if self.z == tag:
            ret.append("z")
        if len(ret) != 1:
            return None
        return ret[0]

    @staticmethod
    def from_str(s: str) -> "SymLabel":
        """Construct symmetry from string.

        .. note:: See also :obj:`STANDARD_SYMMETRY`.
        """
        if s in STANDARD_SYMMETRY:
            return copy.deepcopy(STANDARD_SYMMETRY[s])
        else:
            raise ValueError(f"invalid symmetry: {s}")

    def __str__(self) -> str:
        """
        For human readability, rotations are divided into the following types (names):

        - ``any``: arbitrary rotation is ok;
        - ``cube``: the same symmetry as a cube;
        - ``box``: the same symmetry as a box (flipping along x, y, and z axis);
        - ``none``: no symmetry is provided;
        - ``{x,y,z}-flip``: flip along a single axis;
        - ``{x,y,z}-square-tube``: the same symmetry as a square tube alone the axis;
        - ``{x,y,z}-square-pyramid``: the same symmetry as a pyramid alone the axis;
        - ``{x,y,z}-cylinder``: the same symmetry as a cylinder the axis;
        - ``{x,y,z}-cone``: the same symmetry as a cone the axis.

        """
        c_any = (self.x == "any") + (self.y == "any") + (self.z == "any")
        c_quarter = (
            (self.x == "quarter") + (self.y == "quarter") + (self.z == "quarter")
        )
        c_half = (self.x == "half") + (self.y == "half") + (self.z == "half")

        if self.any or c_any > 1 or (c_any > 0 and c_quarter > 0):  # any rotation is ok
            return "any"

        if c_any == 0:
            if c_quarter > 1:
                return "cube"  # cube group
            elif c_quarter == 0:
                if c_half > 1:
                    return "box"  # cube_flip_group
                else:  # one half or none
                    axis = self.get_only("half")
                    return f"{axis}-flip" if axis is not None else "none"
            else:  # c_quarter == 1
                axis = self.get_only("quarter")
                if c_half > 0:
                    return f"{axis}-square-tube"
                else:
                    return f"{axis}-square-pyramid"
        else:
            assert c_any == 1 and c_quarter == 0
            axis = self.get_only("any")
            if c_half > 0:
                return f"{axis}-cylinder"
            else:
                return f"{axis}-cone"


STANDARD_SYMMETRY = {
    "any": SymLabel(any=True, x="any", y="any", z="any"),
    "cube": SymLabel(any=False, x="quarter", y="quarter", z="quarter"),
    "box": SymLabel(any=False, x="half", y="half", z="half"),
    "none": SymLabel(any=False, x="none", y="none", z="none"),
    "x-flip": SymLabel(any=False, x="half", y="none", z="none"),
    "y-flip": SymLabel(any=False, x="none", y="half", z="none"),
    "z-flip": SymLabel(any=False, x="none", y="none", z="half"),
    "x-square-tube": SymLabel(any=False, x="quarter", y="half", z="half"),
    "y-square-tube": SymLabel(any=False, x="half", y="quarter", z="half"),
    "z-square-tube": SymLabel(any=False, x="half", y="half", z="quarter"),
    "x-square-pyramid": SymLabel(any=False, x="quarter", y="none", z="none"),
    "y-square-pyramid": SymLabel(any=False, x="none", y="quarter", z="none"),
    "z-square-pyramid": SymLabel(any=False, x="none", y="none", z="quarter"),
    "x-cylinder": SymLabel(any=False, x="any", y="half", z="half"),
    "y-cylinder": SymLabel(any=False, x="half", y="any", z="half"),
    "z-cylinder": SymLabel(any=False, x="half", y="half", z="any"),
    "x-cone": SymLabel(any=False, x="any", y="none", z="none"),
    "y-cone": SymLabel(any=False, x="none", y="any", z="none"),
    "z-cone": SymLabel(any=False, x="none", y="none", z="any"),
}
"""
All standard symmetries.

.. doctest::

    >>> for name, sym in cutoop.rotation.STANDARD_SYMMETRY.items():
    ...     assert str(sym) == name, f"name: {name}, sym: {repr(sym)}"

"""

ax = {
    "x": np.array([1, 0, 0]),
    "y": np.array([0, 1, 0]),
    "z": np.array([0, 0, 1]),
}
"""Normalized axis vectors."""

rot90 = {
    "x": _Rotation.from_rotvec(np.pi / 2 * ax["x"]).as_matrix(),
    "y": _Rotation.from_rotvec(np.pi / 2 * ax["y"]).as_matrix(),
    "z": _Rotation.from_rotvec(np.pi / 2 * ax["z"]).as_matrix(),
}
"""90-degree rotation along each axis."""

rot180 = {
    "x": _Rotation.from_rotvec(np.pi * ax["x"]).as_matrix(),
    "y": _Rotation.from_rotvec(np.pi * ax["y"]).as_matrix(),
    "z": _Rotation.from_rotvec(np.pi * ax["z"]).as_matrix(),
}
"""flipping along each axis"""

qtr = {
    "x": [np.eye(3), rot90["x"], rot180["x"], rot90["x"].T],
    "y": [np.eye(3), rot90["y"], rot180["y"], rot90["y"].T],
    "z": [np.eye(3), rot90["z"], rot180["z"], rot90["z"].T],
}
"""All transformations composed by 90-degree rotations along each axis."""

cube_flip_group: "list[ndarray]" = [np.eye(3), rot180["x"], rot180["y"], rot180["z"]]
"""
All 4 rotations of a box, as 3x3 matrices.
That is rotating 180 degrees around x y z, respectively (abelian group).

:meta hide-value:
"""

cube_group: "list[ndarray]" = [
    s @ g @ h
    for s in [np.eye(3), rot90["x"]]
    for g in [rot90["x"] @ rot90["y"], rot90["x"] @ rot90["z"], np.eye(3)]
    for h in cube_flip_group
]
"""
All 24 rotations of a cube (Sym(4)).

The correctness of this implementation lies in:

1. ``cube_flip_group`` is a normal subgroup of alternating group
2. alternating group Alt(4) is a normal subgroup of Sym(4)

Thus we can construct Sym(4) by enumerating all cosets of Alt(4),
which is construct by enumerating all cosets of ``cube_flip_group``.

One can check validity of it by

.. doctest::

    >>> from cutoop.rotation import cube_group
    >>> for i in range(24):
    ...     for j in range(24):
    ...         if i < j:
    ...             diff = cube_group[i] @ cube_group[j].T
    ...             assert np.linalg.norm(diff - np.eye(3)) > 0.1

:meta hide-value:
"""


def rot_diff_theta(rA_3x3: ndarray, rB_Nx3x3: ndarray) -> ndarray:
    """
    Compute the difference angle of one rotation with a series of rotations.

    Note that since `arccos` gets quite steep around 0, the computational loss is
    somehow inevitable (around 1e-4).

    :return: theta (unit: radius) array of length N.
    """
    R = rA_3x3 @ rB_Nx3x3.reshape(-1, 3, 3).transpose(0, 2, 1)
    val = (np.trace(R, axis1=1, axis2=2) - 1) / 2

    if int(os.environ.get("CUTOOP_STRICT_MODE", "0")) == 1:
        assert np.all(np.abs(val) < 1 + 1e-5), f"invalid rotation matrix, cos = {val}"
    theta = np.arccos(np.clip(val, -1, 1))
    return theta


def rot_diff_theta_pointwise(rA_Nx3x3: ndarray, rB_Nx3x3: ndarray) -> ndarray:
    """
    compute the difference angle of two sequences of rotations pointwisely.

    :return: theta (unit: radius) array of length N
    """
    R = np.einsum("ijk,ilk->ijl", rA_Nx3x3, rB_Nx3x3)  # rA @ rB.T
    val = (np.trace(R, axis1=1, axis2=2) - 1) / 2

    if int(os.environ.get("CUTOOP_STRICT_MODE", "0")) == 1:
        assert np.all(np.abs(val) < 1 + 1e-5), f"invalid rotation matrix, cos = {val}"
    theta = np.arccos(np.clip(val, -1, 1))
    return theta


def rot_diff_axis(rA_3x3: ndarray, rB_Nx3x3: ndarray, axis: ndarray) -> ndarray:
    """compute the difference angle where rotation aroud axis is ignored.

    :param axis: One of :obj:`ax`.

    """
    axis = axis.reshape(3)
    y1 = rA_3x3.reshape(3, 3) @ axis
    y2 = rB_Nx3x3.reshape(-1, 3, 3) @ axis
    val = y2.dot(y1) / (np.linalg.norm(y1) * np.linalg.norm(y2, axis=1))

    if int(os.environ.get("CUTOOP_STRICT_MODE", "0")) == 1:
        assert np.all(np.abs(val) < 1 + 1e-5), f"invalid rotation matrix, cos = {val}"
    theta = np.arccos(np.clip(val, -1, 1))
    return theta


def rot_diff_sym(rA_3x3: ndarray, rB_3x3: ndarray, sym: SymLabel) -> float:
    """compute the difference angle (rotation error) with regard of symmetry.

    This function use analytic method to calculate the difference angle,
    which is more accurate than :func:`rot_canonical_sym`.

    :return: the difference angle.
    """

    rA_3x3 = rA_3x3.reshape(3, 3)
    rB_3x3 = rB_3x3.reshape(3, 3)

    c_any = (sym.x == "any") + (sym.y == "any") + (sym.z == "any")
    symname = str(sym)

    if symname == "any":  # any rotation is ok
        return 0
    elif c_any == 0:
        _, theta = rot_canonical_sym(rA_3x3, rB_3x3, sym, return_theta=True)
        return theta
    else:  # c_any == 1 and c_quarter == 0
        assert c_any == 1
        axis = sym.get_only("any")
        vec = ax[axis]
        rB1 = rB_3x3[None, ...]
        if symname.endswith("cylinder"):
            half_axis = "x" if axis == "y" else "y"  # any other axis is ok
            t1 = rot_diff_axis(rA_3x3 @ rot180[half_axis], rB1, vec.reshape(3))[0]
            t2 = rot_diff_axis(rA_3x3, rB1, vec.reshape(3))[0]
            theta = min(t1, t2)
        else:
            theta = rot_diff_axis(rA_3x3, rB1, vec.reshape(3))[0]

    return theta


def rot_canonical_sym(
    rA_3x3: ndarray, rB_3x3: ndarray, sym: SymLabel, split=100, return_theta=False
) -> "ndarray | tuple[ndarray, float]":
    """Find the optimal rotation ``rot`` that minimize ``theta(rA, rB @ rot)``.

    :param rA_3x3: often the ground truth rotation.
    :param rB_3x3: often the predicted rotation.
    :param sym: symmetry label.
    :param split: see :func:`rots_of_sym`.
    :param return_theta: if enabled, a tuple of the ``rot`` and its theta will both
        be returned.

    :returns: the optimal ``rot``
    """

    if str(sym) == "any":
        return rB_3x3.T @ rA_3x3

    rots = rots_of_sym(sym, split=split)
    rBs = rB_3x3 @ rots
    thetas = rot_diff_theta(rA_3x3, rBs)
    index = np.argmin(thetas)
    if return_theta:
        return rots[index], thetas[index]
    else:
        return rots[index]


def rots_of_sym(sym: SymLabel, split=20) -> ndarray:
    """Get a list of rotation group corresponding to the sym label.

    :param split: Set the snap of rotation to ``2 * pi / split`` for continuous
        symmetry.

    :return: ndarray of shape [?, 3, 3] containing a set of rotation matrix.
    """
    snap = 2 * np.pi / split
    symname = str(sym)

    if symname == "any":
        rots = []
        for sx in range(split):
            for sy in range(split):
                for sz in range(split):
                    r = (
                        _Rotation.from_rotvec(ax["x"] * snap * sx)
                        @ _Rotation.from_rotvec(ax["y"] * snap * sy)
                        @ _Rotation.from_rotvec(ax["z"] * snap * sz)
                    )
                    rots.append(r.as_matrix())
        return np.array(rots)  # this is extremely not efficient
    elif symname == "cube":
        return np.array(cube_group)
    elif symname == "box":
        return np.array(cube_flip_group)
    elif symname.endswith("flip"):
        return np.array([np.eye(3), rot180[symname[:1]]])
    elif symname == "none":
        return np.array([np.eye(3)])
    elif symname.endswith("square-tube"):
        return np.array([g @ h for g in qtr[symname[:1]][:2] for h in cube_flip_group])
    elif symname.endswith("square-pyramid"):
        return np.array(qtr[symname[:1]])
    else:
        axis = symname[:1]
        vec = ax[axis]  # norm(vec) == 1
        rots = np.array(
            [_Rotation.from_rotvec(vec * snap * s).as_matrix() for s in range(split)]
        )
        if symname.endswith("cylinder"):
            half_axis = "x" if axis == "y" else "y"  # any other axis is ok
            rots = np.concatenate([rots, rots @ rot180[half_axis]], axis=0)
        # otherwise cone
        return rots
