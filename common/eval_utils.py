"""
Evaluation
==========

This module contains gadgets used to evaluate pose estimation models.

The core structure of evaluation is the dataclass :class:`DetectMatch`, inheriting
from the dataclass :class:`GroundTruth`. In other words, it contains **all information**
required for metrics computation.

.. testsetup::

    import cutoop
    import numpy as np
    from cutoop.data_loader import Dataset
    from cutoop.eval_utils import GroundTruth, DetectMatch, DetectOutput
    from dataclasses import asdict

    objmeta = cutoop.obj_meta.ObjectMetaData.load_json(
        "../../configs/obj_meta.json"
    )

    prefix = "../../misc/sample/0000_"

    image = Dataset.load_color(prefix + "color.png")
    meta = Dataset.load_meta(prefix + "meta.json")
    objects = [obj for obj in meta.objects if obj.is_valid]

    # preparing GroundTruth for evaluation
    gts = []
    for obj in objects:
        objinfo = objmeta.instance_dict[obj.meta.oid]
        gt = GroundTruth(
            # object pose relative to camera
            gt_affine=obj.pose().to_affine()[None, ...],
            # object size under camera space
            gt_size=(
                np.array(obj.meta.scale) * np.array(objinfo.dimensions)
            )[None, ...],
            gt_sym_labels=[objinfo.tag.symmetry],
            gt_class_labels=np.array([objinfo.class_label]),

            # these are optional
            image_path=prefix + "color.png",
            camera_intrinsics=meta.camera.intrinsics,
        )
        gts.append(gt)

    # Concatenate multiple arrays into one.
    gt = GroundTruth.concat(gts)

    np.random.seed(0)
    pred_affine = gt.gt_affine
    pred_size = gt.gt_size * (0.7 + np.random.rand(*gt.gt_size.shape) * 0.6)

    result = DetectMatch.from_gt(gt, pred_affine=pred_affine, pred_size=pred_size)

Evaluation Data Structure
-------------------------

.. autoclass:: DetectMatch
   :inherited-members:
   :special-members: __getitem__

.. autoclass:: DetectOutput
   :inherited-members:

.. autoclass:: GroundTruth
   :inherited-members:

Metrics Data Structure
----------------------

.. autoclass:: Metrics
   :show-inheritance:
   :members:

.. autoclass:: ClassMetrics
   :show-inheritance:
   :members:

.. autoclass:: AUCMetrics
   :show-inheritance:
   :members:

Functions
---------

.. autofunction:: bipart_maximum_match

.. autofunction:: compute_average_precision

.. autofunction:: compute_mask_matches

.. autofunction:: compute_masks_ious

.. autofunction:: group_by_class

"""

from dataclasses import asdict, dataclass
import json
import os
import cv2
import scipy.optimize
import numpy as np
from numpy import ndarray
import numpy.typing as npt
from tqdm import tqdm
from collections import Counter

from .utils import draw_3d_bbox, draw_object_label, draw_pose_axes, pool_map, save_pctxt
from .data_types import CameraIntrinsicsBase
from .transform import sRTdestruct, toAffine
from .rotation import (
    rot_canonical_sym,
    rot_diff_sym,
    rot_diff_theta_pointwise,
)
from . import iou
from .align import create_3d_bbox_pc
from .transform import transform_coordinates_3d


@dataclass
class AUCMetrics:
    auc: float
    """The normalised (divided by the range of thresholds) AUC (or volume under surface) value."""
    xs: "list[list[float]] | list[float] | None"
    """If preserved, this field contains the x coordinates of the curve."""
    ys: "list[list[float]] | list[float] | None"
    """If preserved, this field contains the y coordinates of the curve."""


@dataclass
class ClassMetrics:
    iou_mean: float
    """Average IoU."""
    iou_acc: "list[float]"
    """IoU accuracies over the list of thresholds."""
    iou_auc: "list[AUCMetrics]"
    """IoU (towards 1) AUC over a list of ranges."""
    iou_ap: "list[float] | None"
    """IoU average precision over the list of thresholds."""
    deg_mean: float
    """Average rotation error (unit: degree)."""
    sht_mean: float
    """Average tralsnation error (unit: cm)."""
    pose_acc: "list[float]"
    """Pose accuracies over the list of rotation-translation thresholds."""
    deg_auc: AUCMetrics
    """Rotation (towards 0) AUC."""
    sht_auc: AUCMetrics
    """Translation (towards 0) AUC."""
    pose_auc: "list[AUCMetrics]"
    """Pose error (both towards 0) VUS over a list of ranges."""
    pose_ap: "list[float] | None"
    """Pose error average precision."""

    def __post_init__(self):
        if isinstance(self.deg_auc, dict):
            self.deg_auc = AUCMetrics(**self.deg_auc)
        if isinstance(self.sht_auc, dict):
            self.sht_auc = AUCMetrics(**self.sht_auc)
        if len(self.iou_auc) > 0 and isinstance(self.iou_auc[0], dict):
            self.iou_auc = [AUCMetrics(**x) for x in self.iou_auc]
        if len(self.pose_auc) > 0 and isinstance(self.pose_auc[0], dict):
            self.pose_auc = [AUCMetrics(**x) for x in self.pose_auc]


@dataclass
class Metrics:
    """
    See :meth:`DetectMatch.metrics`.
    """

    class_means: ClassMetrics
    """The mean metrics of all occurred classes."""
    class_metrics: "dict[str, ClassMetrics]"
    """mapping from **class label** to its metrics."""

    def __post_init__(self):
        if isinstance(self.class_means, dict):
            self.class_means = ClassMetrics(**self.class_means)
        for k in self.class_metrics:
            if isinstance(self.class_metrics[k], dict):
                self.class_metrics[k] = ClassMetrics(**self.class_metrics[k])

    def dump_json(self, path: str, mkdir=True):
        """Write metrics to text file as JSON format.

        :param mkdir: enable this flag to automatically create parent directory.
        """
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load_json(path: str) -> "Metrics":
        """Load metrics data from json file"""

        with open(path, "r") as f:
            jsondata = json.load(f)

        return Metrics(**jsondata)


@dataclass
class _GroundTruthBase:
    gt_affine: ndarray
    """Ground truth 3D affine transformation for each instance (Nx4x4), **without scaling**."""
    gt_size: ndarray
    """Ground truth bounding box side lengths (Nx3), a.k.a. ``bbox_side_len`` in 
    :class:`.image_meta.ImageMetaData`.
    """
    gt_sym_labels: npt.NDArray[np.object_]  # ndarray
    """Ground truth symmetry labels (N,) (each element is a :class:`.rotation.SymLabel`).

    .. doctest::

        >>> list(map(str, result.gt_sym_labels))
        ['y-cone', 'x-flip', 'y-cone', 'y-cone', 'none', 'none', 'none', 'none', 'x-flip']

    """
    gt_class_labels: ndarray
    """Ground truth class label of each object."""

    def __len__(self):
        """The number of instances"""
        return len(self.gt_affine)

    def __post_init__(self):  # verify inputs
        if isinstance(self.gt_sym_labels, list):
            self.gt_sym_labels = np.array(self.gt_sym_labels)

    def lens_to_compare(self):
        return [
            len(self.gt_affine),
            len(self.gt_size),
            len(self.gt_sym_labels),
            len(self.gt_class_labels),
        ]


@dataclass
class _GroundTruthDefault:
    image_path: "list[str] | str | None" = None
    """Path to the rgb image (for drawing)."""
    camera_intrinsics: "list[CameraIntrinsicsBase] | CameraIntrinsicsBase | None" = None
    """Camera intrinsics."""

    def lens_to_compare(self):
        nums = []
        if isinstance(self.image_path, list):
            nums.append(len(self.image_path))
        if isinstance(self.camera_intrinsics, list):
            nums.append(len(self.camera_intrinsics))
        return nums


@dataclass
class GroundTruth(_GroundTruthDefault, _GroundTruthBase):
    """
    This dataclass is a subset of :class:`Detectmatch`, which can be constructed
    directly from GT image data before running the inference process.
    """

    def __post_init__(self):  # verify inputs
        nums = _GroundTruthBase.lens_to_compare(
            self
        ) + _GroundTruthDefault.lens_to_compare(self)
        assert max(nums) == min(nums), "data with different length"

    @staticmethod
    def concat(items: "list[GroundTruth]") -> "GroundTruth":
        """Concatenate multiple result into a single one.

        :return: a :class:`GroundTruth` combining all items.
        """
        return GroundTruth(
            gt_affine=np.concatenate([d.gt_affine for d in items]),
            gt_size=np.concatenate([d.gt_size for d in items]),
            gt_sym_labels=np.concatenate([d.gt_sym_labels for d in items]),
            gt_class_labels=np.concatenate([d.gt_class_labels for d in items]),
            image_path=np.concatenate(
                [
                    (
                        d.image_path
                        if type(d.image_path) == list
                        else [d.image_path] * len(d)
                    )
                    for d in items
                ]
            ).tolist(),
            camera_intrinsics=np.concatenate(
                [
                    (
                        d.camera_intrinsics
                        if type(d.camera_intrinsics) == list
                        else [d.camera_intrinsics] * len(d)
                    )
                    for d in items
                ],
            ).tolist(),
        )


@dataclass
class _DetectOutputDefault(_GroundTruthDefault):
    detect_scores: "None | ndarray" = None
    """Confidence output by the detection model (N,)."""
    gt_n_objects: "None | Counter" = None
    """The number of objects annotated in GT (**NOT** the number of detected objects)."""

    def lens_to_compare(self):
        nums = _GroundTruthDefault.lens_to_compare(self)
        if self.detect_scores is not None:
            nums.append(len(self.detect_scores))
        return nums


@dataclass
class DetectOutput(_DetectOutputDefault, _GroundTruthBase):
    def __post_init__(self):  # verify inputs
        nums = _DetectOutputDefault.lens_to_compare(
            self
        ) + _GroundTruthBase.lens_to_compare(self)
        assert max(nums) == min(nums), "data with different length"

    @staticmethod
    def concat(items: "list[DetectOutput]") -> "DetectOutput":
        """Concatenate multiple result into a single one.

        Note that the ``detect_scores`` (``gt_n_objects``) should either be all ``None``
        or all ``ndarray`` (``int``).

        :return: a :class:`DetectOutput` combining all items.
        """
        if all([item.detect_scores is None for item in items]):
            detect_scores = None
        elif all([isinstance(d.detect_scores, ndarray) for d in items]):
            detect_scores = np.concatenate([d.detect_scores for d in items])
        else:
            raise ValueError("invalid prediction scores")

        if all([item.gt_n_objects is None for item in items]):
            gt_n_objects = None
        elif all([isinstance(d.gt_n_objects, Counter) for d in items]):
            gt_n_objects = sum([d.gt_n_objects for d in items])
        else:
            raise ValueError("invalid gt_n_objects")

        return DetectOutput(
            gt_affine=np.concatenate([d.gt_affine for d in items]),
            gt_size=np.concatenate([d.gt_size for d in items]),
            gt_sym_labels=np.concatenate([d.gt_sym_labels for d in items]),
            gt_class_labels=np.concatenate([d.gt_class_labels for d in items]),
            image_path=np.concatenate(
                [
                    (
                        d.image_path
                        if type(d.image_path) == list
                        else [d.image_path] * len(d)
                    )
                    for d in items
                ]
            ).tolist(),
            camera_intrinsics=np.concatenate(
                [
                    (
                        d.camera_intrinsics
                        if type(d.camera_intrinsics) == list
                        else [d.camera_intrinsics] * len(d)
                    )
                    for d in items
                ],
            ).tolist(),
            detect_scores=detect_scores,
            gt_n_objects=gt_n_objects,
        )


@dataclass
class _DetectMatchBase(_GroundTruthBase):
    pred_affine: ndarray
    """Prediction of 3D affine transformation for each instance (Nx4x4)."""
    pred_size: ndarray
    """Prediction of 3D bounding box sizes (Nx3)."""

    def lens_to_compare(self):
        nums = _GroundTruthBase.lens_to_compare(self)
        nums.append(len(self.pred_affine))
        nums.append(len(self.pred_size))
        return nums


@dataclass
class DetectMatch(_DetectOutputDefault, _DetectMatchBase):
    """
    Prediction and its matched ground truth data provided for metric computation.

    Containing 6D pose (rotation, translation, scale) information for ground truth
    and predictions, as well as ground truth symmetry labels.
    """

    def __post_init__(self):  # verify inputs
        nums = _DetectOutputDefault.lens_to_compare(
            self
        ) + _DetectMatchBase.lens_to_compare(self)
        assert max(nums) == min(nums), "data with different length"

    @staticmethod
    def from_gt(
        gt: GroundTruth, pred_affine: ndarray, pred_size: ndarray
    ) -> "DetectMatch":
        """Construct matching result from GT (i. e. use GT detection)."""
        return DetectMatch(
            gt_affine=gt.gt_affine,
            gt_size=gt.gt_size,
            gt_sym_labels=gt.gt_sym_labels,
            gt_class_labels=gt.gt_class_labels,
            pred_affine=pred_affine,
            pred_size=pred_size,
            image_path=gt.image_path,
            camera_intrinsics=gt.camera_intrinsics,
        )

    @staticmethod
    def from_detection(
        detection: DetectOutput, pred_affine: ndarray, pred_size: ndarray
    ) -> "DetectMatch":
        """Construct matching result from the output of a detection model."""
        return DetectMatch(
            gt_affine=detection.gt_affine,
            gt_size=detection.gt_size,
            gt_sym_labels=detection.gt_sym_labels,
            gt_class_labels=detection.gt_class_labels,
            pred_affine=pred_affine,
            pred_size=pred_size,
            image_path=detection.image_path,
            camera_intrinsics=detection.camera_intrinsics,
            detect_scores=detection.detect_scores,
            gt_n_objects=detection.gt_n_objects,
        )

    @staticmethod
    def concat(items: "list[DetectMatch]") -> "DetectMatch":
        """Concatenate multiple result into a single one.

        Note that the :attr:`detect_scores` (:attr:`gt_n_objects`)
        should either be all ``None`` or all ``ndarray`` (``int``).

        :return: a :class:`DetectMatch` combining all items.
        """
        if all([item.detect_scores is None for item in items]):
            detect_scores = None
        elif all([isinstance(d.detect_scores, ndarray) for d in items]):
            detect_scores = np.concatenate([d.detect_scores for d in items])
        else:
            raise ValueError("invalid prediction scores")

        if all([item.gt_n_objects is None for item in items]):
            gt_n_objects = None
        elif all([isinstance(d.gt_n_objects, Counter) for d in items]):
            gt_n_objects = sum([d.gt_n_objects for d in items], Counter())
        else:
            raise ValueError("invalid gt_n_objects")

        gt_affine = np.concatenate([d.gt_affine for d in items])
        gt_size = np.concatenate([d.gt_size for d in items])
        gt_sym_labels = np.concatenate([d.gt_sym_labels for d in items])
        gt_class_labels = np.concatenate([d.gt_class_labels for d in items])
        pred_affine = np.concatenate([d.pred_affine for d in items])
        pred_size = np.concatenate([d.pred_size for d in items])

        return DetectMatch(
            gt_affine=gt_affine,
            gt_size=gt_size,
            gt_sym_labels=gt_sym_labels,
            gt_class_labels=gt_class_labels,
            pred_affine=pred_affine,
            pred_size=pred_size,
            image_path=np.concatenate(
                [
                    (
                        d.image_path
                        if type(d.image_path) == list
                        else [d.image_path] * len(d)
                    )
                    for d in items
                ]
            ).tolist(),
            camera_intrinsics=np.concatenate(
                [
                    (
                        d.camera_intrinsics
                        if type(d.camera_intrinsics) == list
                        else [d.camera_intrinsics] * len(d)
                    )
                    for d in items
                ],
            ).tolist(),
            detect_scores=detect_scores,
            gt_n_objects=gt_n_objects,
        )

    # typo
    def callibrate_rotation(self) -> "DetectMatch":
        return self.calibrate_rotation()

    def calibrate_rotation(self, silent=False) -> "DetectMatch":
        """Calibrate the rotation of pose prediction according to gt symmetry
        labels using :func:`.rotation.rot_canonical_sym`.

        :param silent: enable this flag to hide the tqdm progress bar.

        .. note::
            This function **does not modify** the value in-place. Instead, it produces
            a new calibrated result.

        :return: a new :class:`DetectMatch`.
        """

        if len(self) == 0:
            return self

        _, rAs, _ = sRTdestruct(self.gt_affine)
        sBs, rBs, tBs = sRTdestruct(self.pred_affine)
        crBs = []

        zipped = zip(rAs, rBs, self.gt_sym_labels)
        for rA, rB, sym in (
            zipped if silent else tqdm(zipped, total=len(self), desc="calibrate")
        ):
            crB = rB @ rot_canonical_sym(rA, rB, sym)
            crBs.append(crB)

        crBs = np.array(crBs)
        canonical_pred_affine = toAffine(sBs, crBs, tBs)
        r = DetectMatch(
            gt_affine=self.gt_affine,
            gt_size=self.gt_size,
            gt_sym_labels=self.gt_sym_labels,
            gt_class_labels=self.gt_class_labels,
            pred_affine=canonical_pred_affine,
            pred_size=self.pred_size,
            image_path=self.image_path,
            camera_intrinsics=self.camera_intrinsics,
            detect_scores=self.detect_scores,
            gt_n_objects=self.gt_n_objects,
        )
        r.image_path = self.image_path
        r.camera_intrinsics = r.camera_intrinsics = self.camera_intrinsics
        return r

    def criterion(
        self, computeIOU=True, use_precise_rot_error=False
    ) -> "tuple[ndarray, ndarray, ndarray]":
        """Compute IoUs, rotation differences and translation shifts.
        It is useful if you need to compute other custom metrics based on them.

        When setting ``computeIOU`` to ``False``, it returns an numpy array of zeros.

        :param use_precise_rot_error: Use analytic method for rotation error
            calculation instead of discrete method (enumeration). The results should
            be a little smaller.

        :returns: (ious, theta_degree, shift_cm), where

            - ious: (N,).
            - theta_degree: (N,), unit is degree.
            - shift_cm: (N,), unit is cm.

        .. doctest::

            >>> iou, deg1, sht = result.criterion()
            >>> iou, deg2, sht = result.criterion(use_precise_rot_error=True)
            >>> assert np.abs(deg1 - deg2).max() < 0.05

        """
        sA_N, rA_N, tA_N = sRTdestruct(self.gt_affine)
        sB_N, rB_N, tB_N = sRTdestruct(self.pred_affine)

        # Note that sA and sB are usually 1
        ssa = sA_N[..., None] * self.gt_size
        ssb = sB_N[..., None] * self.pred_size
        tasks = zip(rA_N, tA_N, ssa, rB_N, tB_N, ssb)

        # multithread doesn't optimize
        if computeIOU:
            IoUs = []
            for args in tqdm(tasks, total=len(self), desc="iou"):
                val = iou.compute_3d_bbox_iou(*args)
                IoUs.append(val)
            IoUs = np.array(IoUs)
        else:
            IoUs = np.zeros(len(self))

        if use_precise_rot_error:
            tasks = list(zip(rA_N, rB_N, self.gt_sym_labels))
            theta_degree = pool_map(
                lambda args: rot_diff_sym(*args), tasks, desc="rotation error"
            )
            theta_degree = np.array(theta_degree) / np.pi * 180
        else:  # assume that calibration has been done
            theta_degree = rot_diff_theta_pointwise(rA_N, rB_N) / np.pi * 180

        shift_cm = np.linalg.norm(tA_N - tB_N, axis=1) * 100  # m to cm

        return IoUs, theta_degree, shift_cm

    def metrics(
        self,
        iou_thresholds=[0.25, 0.50, 0.75],
        pose_thresholds=[(5, 2), (5, 5), (10, 2), (10, 5)],
        iou_auc_ranges=[
            (0.25, 1, 0.075),
            (0.5, 1, 0.005),
            (0.75, 1, 0.0025),
        ],
        rot_auc_range=(0, 5, 0.01),
        trans_auc_range=(0, 10, 0.01),
        pose_auc_ranges=[
            ((0, 5, 0.05), (0, 2, 0.02)),
            ((0, 5, 0.05), (0, 5, 0.05)),
            ((0, 10, 0.1), (0, 2, 0.02)),
            ((0, 10, 0.1), (0, 5, 0.05)),
        ],
        auc_keep_curve=False,
        criterion=None,
        use_precise_rot_error=False,
    ) -> Metrics:
        """Compute several pose estimation metrices.

        :param iou_thresholds: threshold list for computing IoU acc. and mAP.
        :param pose_thresholds: rotation-translation threshold list for computing pose acc.
            and mAP.
        :param iou_auc_ranges: list of ranges to compute IoU AUC.
        :param pose_auc_range: degree range and shift range.
        :param auc_keep_curve: enable this flag to output curve points for drawing.
        :param criterion: since the computation of IoU is slow, you may cache the result of
            :meth:`cutoop.eval_utils.DetectMatch.criterion` and provide it here, in exactly
            the same format.
        :param use_precise_rot_error: See :meth:`.eval_utils.DetectMatch.criterion`.

        :returns: the returned format can be formalized as

            - 3D IoU:

              * average (mIoU): per-class average IoU and mean average IoU.
              * accuracy: pre-class IoU accuracy and mean accuracy over a list of thresholds.
              * accuracy AUC: normalised AUC of the IoU's accuracy-thresholds curve.
              * average precision (detected mask, if providing pred_score): per-class
                IoU average precision and mean average precision, using detection
                confidence (mask score) as recall.
            - Pose:

              * average: per-class average rotation and translation error.
              * accuracy: pre-class degree-shift accuracy and mean accuracy over a list of
                thresholds.
              * accuracy AUC: normalised AUC of the rotation's, translation's and pose's
                accuracy-thresholds curve. For _pose_, AUC is generalised to
                "volume under surface".
              * accuracy AUC: AUC of the IoU accuracy-thresholds curve.
              * average precision (detected mask, if providing pred_score): per-class
                degree-shift average precision and mean average precision over a list
                of thresholds, using detection confidence (mask score) as recall.

        """
        if criterion is None:
            ious, degrees, shifts = self.criterion(
                computeIOU=True, use_precise_rot_error=use_precise_rot_error
            )
        else:
            ious, degrees, shifts = criterion

        if self.detect_scores is None:
            labels, cious, cdegrees, cshifts = group_by_class(
                self.gt_class_labels, ious, degrees, shifts
            )
            cscores = [None] * len(labels)
        else:
            labels, cious, cdegrees, cshifts, cscores = group_by_class(
                self.gt_class_labels, ious, degrees, shifts, self.detect_scores
            )

        iou_means = []
        iou_accs = []
        iou_aucs = []
        iou_aps = []
        deg_means = []
        sht_means = []
        pose_accs = []
        deg_aucs = []
        sht_aucs = []
        pose_aucs = []
        pose_aps = []
        class_metrics = {}

        def calc_auc(indicator_fn, range_param):
            # middle point of each small interval
            xs = np.arange(*range_param) + range_param[2] / 2
            ys = np.array([indicator_fn(t) for t in xs])
            return AUCMetrics(
                auc=np.sum(ys * range_param[2]) / (range_param[1] - range_param[0]),
                xs=xs.tolist() if auc_keep_curve else None,
                ys=ys.tolist() if auc_keep_curve else None,
            )

        def calc_vus(indicator_fn, xrange, yrange):
            xs = np.mgrid[slice(*xrange), slice(*yrange)].transpose(1, 2, 0) + np.array(
                [xrange[2] / 2, yrange[2] / 2]
            )
            ys = np.array([[indicator_fn(tx, ty) for (tx, ty) in xraw] for xraw in xs])
            return AUCMetrics(
                auc=np.sum(ys * xrange[2] * yrange[2])
                / (xrange[1] - xrange[0])
                / (yrange[1] - yrange[0]),
                xs=xs.tolist() if auc_keep_curve else None,
                ys=ys.tolist() if auc_keep_curve else None,
            )

        def mean_auc(aucs: "list[AUCMetrics]"):
            return AUCMetrics(
                auc=np.mean([o.auc for o in aucs]),
                xs=(
                    np.mean([o.xs for o in aucs], axis=0).tolist()
                    if auc_keep_curve
                    else None
                ),
                ys=(
                    np.mean([o.ys for o in aucs], axis=0).tolist()
                    if auc_keep_curve
                    else None
                ),
            )

        for ious, degrees, shifts, scores, label in tqdm(
            zip(cious, cdegrees, cshifts, cscores, labels),
            total=len(labels),
            desc="cls metrics",
        ):
            iou_mean = np.mean(ious)
            deg_mean = np.mean(degrees)
            sht_mean = np.mean(shifts)
            iou_acc = [np.mean(ious > t) for t in iou_thresholds]
            iou_auc = [calc_auc(lambda t: np.mean(ious > t), r) for r in iou_auc_ranges]
            pose_acc = [
                np.mean(np.logical_and(degrees < t_deg, shifts < t_sht))
                for (t_deg, t_sht) in pose_thresholds
            ]
            deg_auc = calc_auc(lambda t: np.mean(degrees < t), rot_auc_range)
            sht_auc = calc_auc(lambda t: np.mean(shifts < t), trans_auc_range)
            # compute pose auc (vus)
            pose_auc = [
                calc_vus(
                    lambda t_deg, t_sht: np.mean(
                        np.logical_and(degrees < t_deg, shifts < t_sht)
                    ),
                    xr,
                    yr,
                )
                for (xr, yr) in pose_auc_ranges
            ]

            if scores is None:
                iou_ap = None
                pose_ap = None
            else:
                assert self.gt_n_objects is not None, "gt_n_objects not provided"
                n_gt = self.gt_n_objects[label]  # it must occur
                iou_ap = [
                    compute_average_precision(ious >= t, scores, N_gt=n_gt)
                    for t in iou_thresholds
                ]
                pose_ap = [
                    compute_average_precision(
                        np.logical_and(degrees <= t_deg, shifts <= t_sht),
                        scores,
                        N_gt=n_gt,
                    )
                    for (t_deg, t_sht) in pose_thresholds
                ]

            class_metrics[int(label)] = ClassMetrics(
                iou_mean=iou_mean,
                iou_acc=iou_acc,
                iou_auc=iou_auc,
                iou_ap=iou_ap,
                deg_mean=deg_mean,
                sht_mean=sht_mean,
                pose_acc=pose_acc,
                deg_auc=deg_auc,
                sht_auc=sht_auc,
                pose_auc=pose_auc,
                pose_ap=pose_ap,
            )
            iou_means.append(iou_mean)
            iou_accs.append(iou_acc)
            iou_aucs.append(iou_auc)
            iou_aps.append(iou_ap)
            deg_means.append(deg_mean)
            sht_means.append(sht_mean)
            pose_accs.append(pose_acc)
            deg_aucs.append(deg_auc)
            sht_aucs.append(sht_auc)
            pose_aucs.append(pose_auc)
            pose_aps.append(pose_ap)

        class_means = ClassMetrics(
            iou_mean=np.mean(iou_means, axis=0),
            iou_acc=np.mean(iou_accs, axis=0).tolist(),
            iou_auc=np.apply_along_axis(mean_auc, 0, iou_aucs).tolist(),
            iou_ap=(
                np.mean(iou_aps, axis=0).tolist()
                if self.detect_scores is not None
                else None
            ),
            deg_mean=np.mean(deg_means, axis=0),
            sht_mean=np.mean(sht_means, axis=0),
            pose_acc=np.mean(pose_accs, axis=0).tolist(),
            deg_auc=mean_auc(deg_aucs),
            sht_auc=mean_auc(sht_aucs),
            pose_auc=np.apply_along_axis(mean_auc, 0, pose_aucs).tolist(),
            pose_ap=(
                np.mean(pose_aps, axis=0).tolist()
                if self.detect_scores is not None
                else None
            ),
        )

        return Metrics(
            class_means=class_means,
            class_metrics=class_metrics,
        )

    @staticmethod
    def _draw_image(
        vis_img,
        pred_affine,
        pred_size,
        gt_affine,
        gt_size,
        gt_sym_label,
        camera_intrinsics,
        draw_pred,
        draw_gt,
        draw_label,
        draw_pred_axes_length,
        draw_gt_axes_length,
        thickness,
    ):
        if draw_pred:
            vis_img = draw_3d_bbox(
                vis_img,
                camera_intrinsics,
                pred_affine,
                pred_size,
                color=(255, 0, 0) if draw_gt else None,
                thickness=thickness,
            )
        if draw_pred_axes_length is not None:
            vis_img = draw_pose_axes(
                vis_img,
                camera_intrinsics,
                pred_affine,
                draw_pred_axes_length,
                thickness=thickness,
            )
        if draw_gt:
            vis_img = draw_3d_bbox(
                vis_img,
                camera_intrinsics,
                gt_affine,
                gt_size,
                color=(0, 255, 0) if draw_pred else None,
                thickness=thickness,
            )
        if draw_gt_axes_length is not None:
            vis_img = draw_pose_axes(
                vis_img,
                camera_intrinsics,
                gt_affine,
                draw_gt_axes_length,
                thickness=thickness,
            )
        if draw_label:
            vis_img = draw_object_label(
                vis_img,
                camera_intrinsics,
                gt_affine,
                f"i-{gt_sym_label}",
            )
        return vis_img

    def draw_image(
        self,
        path="./result.png",
        index: "None | int" = None,
        image_root="",
        draw_gt=True,
        draw_pred=True,
        draw_label=True,
        draw_pred_axes_length: "None | float" = None,
        draw_gt_axes_length: "None | float" = None,
        thickness=1,
    ):
        """Draw bbox of gt and prediction on the image. Require
        :attr:`image_path` and :attr:`camera_intrinsics` to be set.

        :param path: output path for rendered image
        :param index: which prediction to draw; set default value None to draw everything on the same image.
        :param image_root: root directory of the image, to which assuming `image_path` stores relative path.
        :param draw_gt: whether to draw gt bbox.
        :param draw_pred: whether to draw predicted bbox.
        :param draw_label: whether to draw symmetry label on the object
        :param draw_pred_axes_length: specify a number to indicate the length of axes of the predicted pose.
        :param draw_gt_axes_length: specify a number to indicate the length of axes of the gt pose.
        :param thickness: specify line thickness.

        .. doctest::

            >>> result.draw_image(
            ...     path='source/_static/gr_1.png'
            ... ) # A
            >>> result.draw_image(
            ...     path='source/_static/gr_2.png',
            ...     index=4,
            ...     draw_label=False,
            ...     draw_pred_axes_length=0.5,
            ... ) # B
            >>> result.draw_image(
            ...     path='source/_static/gr_3.png',
            ...     draw_gt=False,
            ... ) # C
            >>> result.draw_image(
            ...     path='source/_static/gr_4.png',
            ...     draw_pred=False,
            ...     draw_label=False,
            ...     draw_gt_axes_length=0.3,
            ...     thickness=2,
            ... ) # D

        .. testcleanup::

            import os
            import cv2
            import numpy as np
            img1 = cv2.imread('source/_static/gr_1.png')
            img2 = cv2.imread('source/_static/gr_2.png')
            img3 = cv2.imread('source/_static/gr_3.png')
            img4 = cv2.imread('source/_static/gr_4.png')
            h, w, c = img1.shape
            res = np.zeros((h * 2, w * 2, c))
            res[:h, :w, :] = img1
            res[:h, w:, :] = img2
            res[h:, :w, :] = img3
            res[h:, w:, :] = img4
            cv2.imwrite('source/_static/gr_6.png', res)
            os.remove('source/_static/gr_1.png')
            os.remove('source/_static/gr_2.png')
            os.remove('source/_static/gr_3.png')
            os.remove('source/_static/gr_4.png')


        - A(left top): Draw all boxes.
        - B(right top): Draw one object.
        - C(left bottom): Draw predictions.
        - D(right bottom): Draw GT with poses.

        .. image:: _static/gr_6.png

        """
        assert self.image_path is not None, "no image path"
        assert self.camera_intrinsics is not None, "no camera intrinsics"

        if index is None:
            assert len(self) <= 100, "too many instances (> 100)"
            assert isinstance(self.image_path, str) or isinstance(self.image_path, list)
            assert isinstance(
                self.camera_intrinsics, CameraIntrinsicsBase
            ) or isinstance(self.camera_intrinsics, list)
            image_path = (
                self.image_path if type(self.image_path) == str else self.image_path[0]
            )
            camera_intrinsics = (
                self.camera_intrinsics
                if isinstance(self.camera_intrinsics, CameraIntrinsicsBase)
                else self.camera_intrinsics[0]
            )
            vis_img = cv2.imread(os.path.join(image_root, image_path))[:, :, ::-1]
            for i in range(len(self)):
                vis_img = DetectMatch._draw_image(
                    vis_img=vis_img,
                    pred_affine=self.pred_affine[i],
                    pred_size=self.pred_size[i],
                    gt_affine=self.gt_affine[i],
                    gt_size=self.gt_size[i],
                    gt_sym_label=self.gt_sym_labels[i],
                    camera_intrinsics=camera_intrinsics,
                    draw_gt=draw_gt,
                    draw_pred=draw_pred,
                    draw_label=draw_label,
                    draw_gt_axes_length=draw_gt_axes_length,
                    draw_pred_axes_length=draw_pred_axes_length,
                    thickness=thickness,
                )
            cv2.imwrite(path, vis_img[:, :, ::-1])
        else:
            assert isinstance(self.image_path, list)
            assert isinstance(self.camera_intrinsics, list)
            vis_img = cv2.imread(os.path.join(image_root, self.image_path[index]))
            vis_img = vis_img[:, :, ::-1]
            vis_img = DetectMatch._draw_image(
                vis_img=vis_img,
                pred_affine=self.pred_affine[index],
                pred_size=self.pred_size[index],
                gt_affine=self.gt_affine[index],
                gt_size=self.gt_size[index],
                gt_sym_label=self.gt_sym_labels[index],
                camera_intrinsics=self.camera_intrinsics[index],
                draw_gt=draw_gt,
                draw_pred=draw_pred,
                draw_label=draw_label,
                draw_gt_axes_length=draw_gt_axes_length,
                draw_pred_axes_length=draw_pred_axes_length,
                thickness=thickness,
            )
            cv2.imwrite(path, vis_img[:, :, ::-1])

    def save_bbox_pc(self):
        gts = []
        for size, affine in zip(self.gt_size, self.gt_affine):
            bbox = create_3d_bbox_pc(size)
            bbox = transform_coordinates_3d(bbox, affine).T
            gts.append(bbox)
        gts = np.concatenate(gts)
        save_pctxt("./gt_pc.txt", gts)
        preds = []
        for size, affine in zip(self.pred_size, self.pred_affine):
            bbox = create_3d_bbox_pc(size)
            bbox = transform_coordinates_3d(bbox, affine).T
            preds.append(bbox)
        preds = np.concatenate(preds)
        save_pctxt("./pred_pc.txt", preds)

    def __getitem__(self, index) -> "DetectMatch":
        """Use slice or integer index to get a subset sequence of data.

        Note that :attr:`gt_n_objects` would be lost.

        .. doctest::

            >>> result[1:3]
            DetectMatch(gt_affine=array([[[ 0.50556856,  0.06278258,  0.86049914, -0.2933575 ],
                    [-0.57130486, -0.72301346,  0.3884099 ,  0.5845589 ],
                    [ 0.6465379 , -0.68797517, -0.3296649 ,  1.7387577 ],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]],
            <BLANKLINE>
                   [[ 0.38624182,  0.00722773, -0.92236924,  0.02853431],
                    [ 0.6271931 , -0.735285  ,  0.25687516,  0.6061586 ],
                    [-0.6763476 , -0.67771953, -0.28853098,  1.6325369 ],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]]],
                  dtype=float32), gt_size=array([[0.34429058, 0.00904833, 0.19874577],
                   [0.32725419, 0.1347834 , 0.32831856]]), gt_sym_labels=array([SymLabel(any=False, x='half', y='none', z='none'),
                   SymLabel(any=False, x='none', y='any', z='none')], dtype=object), gt_class_labels=array([48,  9]), pred_affine=array([[[ 0.50556856,  0.06278258,  0.86049914, -0.2933575 ],
                    [-0.57130486, -0.72301346,  0.3884099 ,  0.5845589 ],
                    [ 0.6465379 , -0.68797517, -0.3296649 ,  1.7387577 ],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]],
            <BLANKLINE>
                   [[ 0.38624182,  0.00722773, -0.92236924,  0.02853431],
                    [ 0.6271931 , -0.735285  ,  0.25687516,  0.6061586 ],
                    [-0.6763476 , -0.67771953, -0.28853098,  1.6325369 ],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]]],
                  dtype=float32), pred_size=array([[0.3535623 , 0.00863385, 0.21614327],
                   [0.31499929, 0.1664661 , 0.41965601]]), image_path=['../../misc/sample/0000_color.png', '../../misc/sample/0000_color.png'], camera_intrinsics=[CameraIntrinsicsBase(fx=1075.41, fy=1075.69, cx=631.948, cy=513.074, width=1280, height=1024), CameraIntrinsicsBase(fx=1075.41, fy=1075.69, cx=631.948, cy=513.074, width=1280, height=1024)], detect_scores=None, gt_n_objects=None)
        """
        if isinstance(index, bool):
            raise ValueError("invalid boolean type index")
        if isinstance(index, int):
            index = [index]  # singleton
        return DetectMatch(
            gt_affine=self.gt_affine[index],
            gt_size=self.gt_size[index],
            gt_sym_labels=self.gt_sym_labels[index],
            gt_class_labels=self.gt_class_labels[index],
            pred_affine=self.pred_affine[index],
            pred_size=self.pred_size[index],
            image_path=(
                np.array(self.image_path)[index].tolist()
                if type(self.image_path) == list
                else self.image_path
            ),
            camera_intrinsics=(
                np.array(self.camera_intrinsics)[index].tolist()
                if type(self.camera_intrinsics) == list
                else self.camera_intrinsics
            ),
            detect_scores=(
                self.detect_scores[index] if self.detect_scores is not None else None
            ),
            gt_n_objects=None,
        )


def compute_masks_ious(masks1_HxWxN: ndarray, masks2_HxWxM: ndarray) -> ndarray:
    """
    Computes IoU overlaps between each pair of two sets of masks.

    masks1, masks2: [Height, Width, instances]

    Masks can be float arrays, the value > 0.5 is considered True, otherwise False.

    Returns: N x M ious
    """

    assert len(masks1_HxWxN.shape) == 3, "masks should be 3d array"
    assert len(masks2_HxWxM.shape) == 3, "masks should be 3d array"

    # flatten masks
    masks1 = np.reshape(masks1_HxWxN > 0.5, (-1, masks1_HxWxN.shape[-1])).astype(
        np.float32
    )
    masks2 = np.reshape(masks2_HxWxM > 0.5, (-1, masks2_HxWxM.shape[-1])).astype(
        np.float32
    )
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    # avoid divide by zero error
    overlaps = intersections * ((1 / union) if np.all(union > 1e-7) else 0)

    return overlaps


def bipart_maximum_match(
    weights_NxM: ndarray, min_weight=0.0
) -> "tuple[ndarray, ndarray]":
    """Find a maximum matching of a bipart graph match with approximately maximum weights
    using `linear_sum_assignment <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>`_.

    :param weights_NxM: weight array of the bipart graph
    :param min_weight: (int): minimum valid weight of matched edges

    :return: (l_matches, r_matches)

        - l_match: array of length N containing matching index in r, -1 for not matched
        - r_match: array of length M containing matching index in l, -1 for not matched

    """

    assert len(weights_NxM.shape) == 2, "weights should be a 2D array"
    nl, nr = weights_NxM.shape

    # Calculates a maximum weight matching using Jonker-Volgenant algorithm
    #
    # DF Crouse. On implementing 2D rectangular assignment algorithms.
    # IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679-1696,
    # August 2016, DOI:10.1109/TAES.2016.140952
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(weights_NxM, True)

    # filter match result
    valid_ind = weights_NxM[row_ind, col_ind] >= min_weight
    row_ind = row_ind[valid_ind]
    col_ind = col_ind[valid_ind]

    l_match = -1 * np.ones(nl, dtype=int)
    r_match = -1 * np.ones(nr, dtype=int)
    l_match[row_ind] = col_ind
    r_match[col_ind] = row_ind

    return l_match, r_match


def compute_mask_matches(
    gt_masks_HxWxN: ndarray,
    gt_classes_N: ndarray,
    pred_scores_M,
    pred_masks_HxWxM: ndarray,
    pred_classes_M: "ndarray | None" = None,
    iou_threshold=0.5,
):
    """
    Finds matches between prediction and ground truth instances. Two matched objects
    must belong to the same class. Under this restriction, the match with higher IoU
    is of greater chance to be selected.

    Scores are used to sort predictions.

    :returns: (gt_match, pred_match, overlaps)

        - gt_match: 1-D array. For each GT mask it has the index of the matched
          predicted mask, otherwise it is marked as -1.
        - pred_match: 1-D array. For each predicted mask, it has the index of the
          matched GT mask, otherwise it is marked as -1.
        - overlaps: [M, N] IoU overlaps (negative values indicate distinct classes).

    """

    assert len(gt_masks_HxWxN.shape) == 3, "masks should be 3d array"
    assert len(pred_masks_HxWxM.shape) == 3, "masks should be 3d array"

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores_M)[::-1]
    pred_masks_HxWxM = pred_masks_HxWxM[..., indices]

    inv_indices = np.arange(indices.shape[0])
    inv_indices[indices] = inv_indices.copy()

    overlaps = compute_masks_ious(pred_masks_HxWxM, gt_masks_HxWxN)
    overlaps = overlaps[inv_indices]

    if pred_classes_M is not None:
        class_match = pred_classes_M[..., None] == gt_classes_N[None, ...]
        overlaps[~class_match] = -1

    pred_match, gt_match = bipart_maximum_match(overlaps, min_weight=iou_threshold)

    return gt_match, pred_match, overlaps


def compute_average_precision(
    pred_indicator_M: ndarray, pred_scores_M: ndarray, N_gt: "None | int" = None
) -> float:
    """
    Calculate average precision (AP).

    The detailed algorithm is

    1.  Sort ``pred_indicator_M`` by ``pred_scores_M`` in decreasing order, and compute precisions as

        .. math:: p_i = \\frac{1}{i} \\sum_{j=1}^i \\texttt{pred\\_indicator\\_M[i]}

        .. note::

            According to `this issue <https://github.com/Lightning-AI/torchmetrics/issues/1966>`_,
            when encountering equal scores for different indicator values, the order of them affect
            the final result of AP. To eliminate multiple possible outputs, we add a second sorting
            key --- sort indicator value from high to low when they possess equal scores.

    2.  Compute recalls by (:math:`0 \\le r_i \\le 1, r_0 = 0`)

        .. math:: r_i = \\frac{1}{\\texttt{N\\_gt}} \\sum_{j=1}^i \\texttt{pred\\_indicator\\_M[i]}

    3.  Suffix maximize the precisions:

        .. math:: P_i = \\max_{j \ge i} \\{ p_j \\}

    4.  Compute average precision (`ref <https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_)

        .. math:: AP = \\sum_{i = 1}^n (r_i - r_{i - 1}) \\times P_i

    Parameters:

    :param pred_indicator_M: A 1D 0-1 array, 0 for false positive, 1 for true positive.
    :param pred_scores_M: Confidence (e.g. IoU) of the prediction, between `[0, 1]`.
    :param N_gt: Number of ground truth instances. If not provided, it is set to M.

    :returns: the AP value. If M == 0, return NaN.

    .. doctest::

        >>> inds = np.array([1, 0, 1, 0, 1, 0, 0])
        >>> scores = np.array([1, 1, 1, 1, 1, 1, 1])
        >>> float(cutoop.eval_utils.compute_average_precision(inds, scores))
        0.4285714328289032

    """
    if len(pred_indicator_M) == 0:
        return np.nan
    if N_gt is None:
        N_gt = len(pred_indicator_M)

    assert (
        pred_indicator_M.shape == pred_scores_M.shape
    ), f"{pred_indicator_M.shape} != {pred_scores_M.shape}"

    # Sort the scores from high to low. For equal scores,
    score_indices = np.argsort(-pred_indicator_M.astype(np.int32))
    pred_indicator_M = pred_indicator_M[score_indices]
    pred_scores_M = pred_scores_M[score_indices]

    score_indices = np.argsort(-pred_scores_M)
    pred_indicator_M = pred_indicator_M[score_indices]

    # print("indicator", pred_indicator_M)

    # precision for each rank threshold
    precisions = np.cumsum(pred_indicator_M) / (np.arange(len(pred_indicator_M)) + 1)
    # rank thresholds
    recalls = np.cumsum(pred_indicator_M).astype(np.float32) / N_gt
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # print("precision", precisions)
    # print("recall", recalls)

    # FIXME: how does it make sense?
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    # precisions = np.maximum.accumulate(precisions[::-1])[::-1]  # suffix maximum

    # != still work for float numbers
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    # print("indices", indices)
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap


def group_by_class(class_labels: ndarray, *values: ndarray):
    """Separate the original sequence into several subsequences, of which each
    belongs to a unique class label. The corresponding class label is also returned.

    :return: a tuple ``(*sequences, labels)`` where

        -   ``sequences`` is a list of :math:`N`
            lists of :math:`C` ndarrays, where :math:`N` is the number of ``values``, :math:`C`
            is the number of occurred classes.
        -   ``labels`` is a ndarray of length :math:`C`.

    .. doctest::

        >>> from cutoop.eval_utils import group_by_class
        >>> group_by_class(np.array([0, 1, 1, 1, 1, 0, 1]), np.array([0, 1, 4, 2, 8, 5, 7]))
        (array([0, 1]), [array([0, 5]), array([1, 4, 2, 8, 7])])

    """

    seqs = []
    for value in values:
        labels = np.unique(class_labels)
        inds = labels[..., None] == class_labels[None, ...]
        seq = []
        for ind in inds:
            seq.append(value[ind])
        seqs.append(seq)
    return (labels, *seqs)
