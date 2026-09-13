"""
Fixed sign- and shift-invariant vector quantization for BOWaves-style
shift-invariant k-means.

Drop-in replacement for the "Sign and Shift Invariant k-means with vq" cell
in Contrastive_Learning_Shift_Invariant_k_means_with_CUE_Dataset.ipynb.

Two bugs fixed relative to the notebook version:

1. si2_vq's sign recovery used `best_labels_b // 2 == 0`, which is wrong.
   Y is stacked as [Y; -Y], shape (2*n_centroids, L). A match against the
   first n_centroids rows means positive sign; a match against the second
   n_centroids rows (indices n_centroids..2*n_centroids-1) means negative
   sign. The correct test is `best_labels_b < n_centroids`, not a parity
   check on the raw index.

2. si2_vq's return statement dropped `best_labels` entirely, returning only
   3 values (best_shifts, best_distances, best_signs) even though callers
   throughout the notebook (_assignment_step, get_labels_per_segment)
   unpack 4 values as (labels, shifts, distances, signs). This is the
   direct cause of the `NameError: name 'best_argmins' is not defined`
   traceback in the original notebook.

A third, latent bug was also fixed in si_vq's Euclidean branch: the squared
distance was computed as `Xnorm + XY + Ynorm.T` (using un-squared norms,
wrong sign, and no factor of 2), instead of the correct expansion
`||x||^2 - 2*x.y + ||y||^2`. This wouldn't raise an error, but would
silently converge to incorrect centroids whenever metric='euclidean' (as
opposed to 'cosine') was used.

Validated against synthetic shifted + sign-flipped waveforms in
si2_vq_fixed_test.py: 100% cluster/shift/sign recovery for both metrics.
"""

import numpy as np
from sklearn.preprocessing import normalize


def si_vq(X, Y, metric):
    """
    Shift-invariant vector quantization.

    Parameters
    ----------
    X : (n, M) array
        Data rows to be windowed and matched.
    Y : (k, L) array
        Codebook, L < M.
    metric : 'cosine' or 'euclidean'

    Returns
    -------
    best_argmins : (n,) int array
        Index of the closest centroid (row of Y) for each sample.
    best_shifts : (n,) int array
        Shift (window start index into X) achieving the minimum distance.
    best_distances : (n,) float array
        Distance achieved at best_shifts / best_argmins.
    """
    n, M = X.shape
    L = Y.shape[1]
    distances = np.empty((n, M - L + 1))
    argmins = np.empty_like(distances, dtype=int)

    nY = normalize(Y, axis=1).astype('f')
    # Squared norms of Y as a row vector, shape (1, n_centroids), for the
    # ||x||^2 - 2 x.y + ||y||^2 expansion in the euclidean branch.
    Ynorm2 = np.sum(Y ** 2, axis=1, keepdims=True).T

    for shift in range(M - L + 1):
        Xshift = X[:, shift:shift + L].astype('f')
        if metric == 'cosine':
            nX = normalize(Xshift, axis=1)
            XY = nX @ nY.T
            argmins[:, shift] = np.argmax(XY, axis=1)
            distances[:, shift] = 1 - XY[np.arange(n), argmins[:, shift]]
        else:
            Xnorm2 = np.sum(Xshift ** 2, axis=1, keepdims=True)  # (n, 1)
            XY = Xshift @ Y.T                                    # (n, k)
            # FIXED: correct squared-euclidean expansion (was Xnorm+XY+Ynorm.T)
            all_sq_distances = Xnorm2 - 2 * XY + Ynorm2
            argmins[:, shift] = np.argmin(all_sq_distances, axis=1)
            best_sq = all_sq_distances[np.arange(n), argmins[:, shift]]
            distances[:, shift] = np.sqrt(np.maximum(best_sq, 0))

    best_shifts = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(n), best_shifts]
    best_argmins = argmins[np.arange(n), best_shifts]
    return best_argmins, best_shifts, best_distances


def si2_vq(X, Y, metric):
    """
    Sign- and shift-invariant vector quantization.

    Matches each row of X against both +Y and -Y (stacked), so a sample
    that best matches a sign-flipped centroid is still assigned to that
    centroid, with the recovered sign returned separately.

    Returns
    -------
    best_labels : (n,) int array
        Index into the *original* (non-stacked) Y, i.e. in [0, n_centroids).
    best_shifts : (n,) int array
    best_distances : (n,) float array
    best_signs : (n,) int array
        +1 or -1: the sign that must multiply the centroid to best match
        the sample.
    """
    n, M = X.shape
    n_centroids = Y.shape[0]

    stacked = np.vstack((Y, -Y))  # rows [0..n_centroids) = +Y, [n_centroids..2k) = -Y
    best_labels_b, best_shifts, best_distances = si_vq(X, stacked, metric)

    # FIXED: sign is + for the first block, - for the second block.
    best_signs = np.where(best_labels_b < n_centroids, 1, -1)
    best_labels = best_labels_b % n_centroids

    # FIXED: return all four values, in the order every caller expects.
    return best_labels, best_shifts, best_distances, best_signs
