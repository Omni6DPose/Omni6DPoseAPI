"""
Point Cloud Alignment
=====================
"""

import numpy as np
from .bbox import create_3d_bbox_pc


def estimateSimilarityUmeyama(SourceHom, TargetHom):
    """
    Compute `OutTransform` s.t. `OutTransform` @ `SourceHom` approx. `TargetHom`
    Copy of original paper is at: `umeyama <http://web.stanford.edu/class/cs273/refs/umeyama.pdf>`_

    :param SourceHom: (4, N)
    :param TargetHom: (4, N)

    :return: scalar Scale, (3, 3) Rotation, (3,) Translation, (4, 4) OutTransform
    """
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = (
        SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    )
    CenteredTarget = (
        TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()
    )
    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints
    if np.isnan(CovMatrix).any():
        print("nPoints:", nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError("There are NANs in the input.")

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    # rotation
    Rotation = np.matmul(U, Vh)
    # scale
    varP = np.var(SourceHom[:3, :], axis=1).sum()
    Scale = 1 / varP * np.sum(D)
    # translation
    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(
        Scale * Rotation.T
    )
    # transformation matrix
    OutTransform = np.identity(4)
    OutTransform[:3, :3] = Scale * Rotation
    OutTransform[:3, 3] = Translation

    return Scale, Rotation, Translation, OutTransform


def estimateSimilarityTransform(
    source_Nx3: np.ndarray, target_Nx3: np.ndarray, verbose=False
):
    """Compute an affine `OutTransform` from `source_Nx3` to `target_Nx3`,
    Add RANSAC algorithm to account for outliers.

    Copying from `object-deformnet <https://github.com/mentian/object-deformnet/blob/a2dcdb87dd88912c6b51b0f693443212fde5696e/lib/align.py#L44>`_

    :returns: scalar Scale, (3, 3) Rotation, (3,) Translation, (4, 4) OutTransform
    """
    assert (
        source_Nx3.shape[0] == target_Nx3.shape[0]
    ), "Source and Target must have same number of points."
    SourceHom = np.transpose(np.hstack([source_Nx3, np.ones([source_Nx3.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target_Nx3, np.ones([target_Nx3.shape[0], 1])]))
    # Auto-parameter selection based on source heuristics
    # Assume source is object model or gt nocs map, which is of high quality
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = (
        SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    )
    SourceDiameter = 2 * np.amax(np.linalg.norm(CenteredSource, axis=0))
    InlierT = SourceDiameter / 10.0  # 0.1 of source diameter
    maxIter = 128
    confidence = 0.99

    if verbose:
        print("Inlier threshold: ", InlierT)
        print("Max number of iterations: ", maxIter)

    BestInlierRatio = 0
    BestInlierIdx = np.arange(nPoints)
    for i in range(0, maxIter):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(nPoints, size=5)
        Scale, _, _, OutTransform = estimateSimilarityUmeyama(
            SourceHom[:, RandIdx], TargetHom[:, RandIdx]
        )
        PassThreshold = Scale * InlierT  # propagate inlier threshold to target scale
        Diff = TargetHom - np.matmul(OutTransform, SourceHom)
        ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
        InlierIdx = np.where(ResidualVec < PassThreshold)[0]
        nInliers = InlierIdx.shape[0]
        InlierRatio = nInliers / nPoints
        # update best hypothesis
        if InlierRatio > BestInlierRatio:
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if verbose:
            print("Iteration: ", i)
            print("Inlier ratio: ", BestInlierRatio)
        # early break
        if (1 - (1 - BestInlierRatio**5) ** i) > confidence:
            break

    if BestInlierRatio < 0.1:
        print("[ WARN ] - Something is wrong. Small BestInlierRatio: ", BestInlierRatio)
        return None, None, None, None

    SourceInliersHom = SourceHom[:, BestInlierIdx]
    TargetInliersHom = TargetHom[:, BestInlierIdx]
    Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(
        SourceInliersHom, TargetInliersHom
    )

    if verbose:
        print("BestInlierRatio:", BestInlierRatio)
        print("Rotation:\n", Rotation)
        print("Translation:\n", Translation)
        print("Scale:", Scale)

    return Scale, Rotation, Translation, OutTransform
