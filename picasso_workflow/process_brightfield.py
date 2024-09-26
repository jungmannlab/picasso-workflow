#!/usr/bin/env python
"""
Module Name: process_brightfield.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: This module implements functionality for processing non-DNA-PAINT
images.
"""
import logging
from moviepy.editor import ImageSequenceClip
from imageio import imsave  # package is dependency of moviepy
import numpy as np

logger = logging.getLogger(__name__)


def adjust_contrast(img, min_quantile, max_quantile):
    """Adjusts contrast of a 2D grayscale image, and returns it
    compatible with moviepy.
    Args:
        img : 2D np array (x, y)
            the grayscale image data
        min_quantile : float
            the quantile below which everything should be displayed
            as black
        max_quantile : float
            the quantile above which everythin should be displayed
            as white
    Returns:
        adjusted_image : 3D np array (x, y, 3)
            the moviepy-compatible grayscale image with identical values
            for RGB, and a bit depth of 8.
    """
    min_val = np.quantile(img, min_quantile)
    max_val = np.quantile(img, max_quantile)
    img = img.astype(np.float32) - min_val
    img = img * 255 / (max_val - min_val)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return np.rollaxis(np.array([img, img, img], dtype=np.uint8), 0, 3)


def save_movie(fname, movie, min_quantile=0, max_quantile=1, fps=1):
    """Save a grayscale movie to file.
    Args:
        fname : str
            the file name to save
        movie : 3D np array (x,y,t)
            the grayscale movie to save
        min_quantile : float, default: 0
            the quantile below which pixels are shown black
        max_quantile : float, default: 1
            the quantile above which pixels are shown white
        fps : float
            the playback speed in frames per second
    """
    adjusted_images = [
        adjust_contrast(frame, min_quantile, max_quantile)[..., np.newaxis]
        for frame in movie
    ]

    # Create movie file
    clip = ImageSequenceClip(adjusted_images, fps=fps)
    clip.write_videofile(fname, verbose=False)  # , codec='mpeg4')


def save_frame(pathname, frame, min_quantile=0, max_quantile=1):
    """Save a grayscale frame to png
    Args:
        pathname : str
            the file name to save
        frame : 2D np array (x,y)
            the frame to save
        min_quantile : float, default: 0
            the quantile below which pixels are shown black
        max_quantile : float, default: 1
            the quantile above which pixels are shown white
    """
    logger.debug(frame.shape)
    adjusted_frame = adjust_contrast(
        frame, min_quantile, max_quantile
    )  # [..., np.newaxis]
    logger.debug(adjusted_frame.shape)
    imsave(pathname, adjusted_frame)
