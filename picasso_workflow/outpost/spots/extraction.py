"""Spot extraction -- cut/normalise pixel boxes around localizations.

Extracted from picasso_outpost (monolith split S3b). `keep` -- localization
spot-cutting for re-fitting; ``_cut_spots_byrandomframe`` is a deliberate
random-frame sampler (not a dedup of ``localize.get_spots``).
"""

import numpy as np

from picasso_workflow._picasso_api import get_internal


def get_spots(movie, identifications, box, camera_info):
    """Cut spot boxes from a movie and convert them to photons.

    Parameters
    ----------
    movie : array-like
        The image data ``(t, x, y)``.
    identifications : recarray
        Identified spot positions (``frame``, ``x``, ``y``).
    box : int
        The (odd) box size to cut around each spot.
    camera_info : dict
        Camera parameters for the photon conversion.

    Returns
    -------
    numpy.ndarray
        The cut spots in photons.
    """
    spots = _cut_spots(movie, identifications, box)
    return get_internal("_to_photons")(spots, camera_info)


def _cut_spots(movie, ids, box):
    """Cut ``box``-sized spots from ``movie`` at the identified positions."""
    N = len(ids.frame)
    spots = np.zeros((N, box, box), dtype=movie.dtype)
    spots = _cut_spots_byrandomframe(
        movie, ids.frame, ids.x, ids.y, box, spots
    )
    return spots


def _cut_spots_byrandomframe(movie, ids_frame, ids_x, ids_y, box, spots):
    """Cut spots out of a movie by (unsorted) frame.

    Parameters
    ----------
    movie : AbstractPicassoMovie
        The image data ``(t, x, y)``.
    ids_frame, ids_x, ids_y : 1D array
        Spot positions in the image data, length = number of spots
        identified.
    box : int
        The (odd) cut-spot box size.
    spots : 3D array
        Pre-allocated output array of shape ``(k, box, box)``.

    Returns
    -------
    3D array
        The ``spots`` array filled with image data.
    """
    r = int(box / 2)
    for j, (fr, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        frame = movie[fr]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return spots


def normalize_spot(spot, maxval=255, dtype=np.uint8):
    """Rescale a spot to ``[0, maxval]`` and cast to ``dtype``.

    Parameters
    ----------
    spot : numpy.ndarray
        The spot image.
    maxval : int, optional
        The maximum value after rescaling. Default is 255.
    dtype : numpy.dtype, optional
        The output dtype. Default is ``np.uint8``.

    Returns
    -------
    numpy.ndarray
        The normalized spot.
    """
    # logger.debug('spot input: ' + str(spot))
    sp = spot - np.min(spot)
    imgmax = np.max(sp)
    imgmax = 1 if imgmax == 0 else imgmax
    sp = sp.astype(np.float32) / imgmax * maxval
    # logger.debug('spot output: ' + str(sp.astype(dtype)))
    return sp.astype(dtype)
