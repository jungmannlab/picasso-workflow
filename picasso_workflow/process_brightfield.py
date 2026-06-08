#!/usr/bin/env python
"""Process non-DNA-PAINT images (e.g. brightfield) for documentation.

Provides contrast adjustment and moviepy/PNG export helpers for grayscale
frames and movies.

Author: Heinrich Grabmayr
Initial date: March 7, 2024
"""

from __future__ import annotations

# import logging
from loguru import logger
from moviepy.editor import ImageSequenceClip
from imageio import imsave  # package is dependency of moviepy
import numpy as np

# logger = logging.getLogger(__name__)


def adjust_contrast(
    img: np.ndarray, min_quantile: float, max_quantile: float
) -> np.ndarray:
    """Adjust the contrast of a 2D grayscale image for moviepy.

    Parameters
    ----------
    img : np.ndarray
        The grayscale image data, shape ``(x, y)``.
    min_quantile : float
        The quantile below which everything is displayed as black.
    max_quantile : float
        The quantile above which everything is displayed as white.

    Returns
    -------
    np.ndarray
        The moviepy-compatible grayscale image, shape ``(x, y, 3)`` with
        identical values for R, G and B and a bit depth of 8.
    """
    min_val = np.quantile(img, min_quantile)
    max_val = np.quantile(img, max_quantile)
    img = img.astype(np.float32) - min_val
    img = img * 255 / (max_val - min_val)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return np.rollaxis(np.array([img, img, img], dtype=np.uint8), 0, 3)


def save_movie(
    fname: str,
    movie: np.ndarray,
    min_quantile: float = 0,
    max_quantile: float = 1,
    fps: float = 1,
) -> None:
    """Save a grayscale movie to file.

    Parameters
    ----------
    fname : str
        The file name to save.
    movie : np.ndarray
        The grayscale movie to save, shape ``(x, y, t)``.
    min_quantile : float, optional
        The quantile below which pixels are shown black. Default is 0.
    max_quantile : float, optional
        The quantile above which pixels are shown white. Default is 1.
    fps : float, optional
        The playback speed in frames per second. Default is 1.
    """
    adjusted_images = [
        adjust_contrast(frame, min_quantile, max_quantile)[..., np.newaxis]
        for frame in movie
    ]

    # Create movie file
    clip = ImageSequenceClip(adjusted_images, fps=fps)
    clip.write_videofile(fname, verbose=False)  # , codec='mpeg4')


def save_frame(
    pathname: str,
    frame: np.ndarray,
    min_quantile: float = 0,
    max_quantile: float = 1,
) -> None:
    """Save a grayscale frame to a PNG file.

    Parameters
    ----------
    pathname : str
        The file name to save.
    frame : np.ndarray
        The frame to save, shape ``(x, y)``.
    min_quantile : float, optional
        The quantile below which pixels are shown black. Default is 0.
    max_quantile : float, optional
        The quantile above which pixels are shown white. Default is 1.
    """
    logger.debug(frame.shape)
    adjusted_frame = adjust_contrast(
        frame, min_quantile, max_quantile
    )  # [..., np.newaxis]
    logger.debug(adjusted_frame.shape)
    imsave(pathname, adjusted_frame)
