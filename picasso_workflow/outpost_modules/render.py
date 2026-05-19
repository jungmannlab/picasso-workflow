#!/usr/bin/env python
"""
Module Name: render.py
Author: Heinrich Grabmayr
Initial Date: Jan 22, 2025
Description: wrapping picasso.gui.render and pricasso.render
    for matplotlib plotting
"""
# import logging
from loguru import logger

from picasso import render
import matplotlib.pyplot as plt
import numpy as np
import colorsys


# logger = logging.getLogger(__name__)


def render_scene(kwargs, locs, viewport=None):
    """
    Returns QImage with rendered localizations.

    Parameters
    ----------
    autoscale : boolean (default=False)
        True if optimally adjust contrast
    locs : list of np.rec.array
        the channel locs
    viewport : tuple (default=None)
        Viewport to be rendered. If None, takes current viewport

    Returns
    -------
    bgra : (M, N, 4)
        BGRalpha image; 8 bit
    """

    if viewport is not None:
        kwargs["viewport"] = viewport
    n_group_colors = kwargs.get("n_group_colors", 8)
    cmap = kwargs.get("cmap", "magma")

    n_channels = len(locs)
    # render single or multi channel data
    if n_channels == 1:
        bgra = render_single_channel(
            kwargs, locs[0], n_group_colors=n_group_colors, cmap=cmap
        )
    else:
        bgra = render_multi_channel(kwargs, locs)

    # add alpha channel (no transparency)
    bgra[:, :, 3].fill(255)

    return bgra


def render_single_channel(kwargs, locs, n_group_colors=8, cmap="magma"):
    """
    Renders single channel localizations.

    Calls render_multi_channel in case of clustered or picked locs,
    rendering by property)

    Parameters
    ----------
    kwargs : dict
        Contains blur method, etc. See self.get_render_kwargs
    autoscale : boolean (default=False)
        True if optimally adjust contrast
    use_cache : boolean (default=False)
        True if use stored image
    cache : boolena (default=True)
        True if save image

    Returns
    -------
    np.array
        8 bit array with 4 channels (rgb and alpha)
    """
    # if locs have group identity (e.g. clusters)
    if hasattr(locs, "group") and locs.group.size:
        group_colors = get_group_color(locs, n_group_colors)
        locs = [locs[group_colors == _] for _ in range(n_group_colors)]
        return render_multi_channel(kwargs, locs=locs)
    n_locs, image = render.render(locs, **kwargs)

    # adjust contrast and convert to 8 bits
    image = scale_contrast([image])[0]
    image = to_8bit(image)

    # paint locs using the colormap of choice (Display Settings
    # Dialog)
    cmap = np.uint8(np.round(255 * plt.get_cmap(cmap)(np.arange(256))))

    # return a 4 channel (rgb and alpha) array
    Y, X = image.shape
    bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
    bgra[..., 0] = cmap[:, 2][image]
    bgra[..., 1] = cmap[:, 1][image]
    bgra[..., 2] = cmap[:, 0][image]

    return bgra


def render_multi_channel(
    kwargs,
    locs,
):
    """
    Renders and paints multichannel localizations.

    Also used when other multi-color data is used (clustered or
    picked locs, render by property)

    Parameters
    ----------
    kwargs : dict
        Contains blur method, etc
        oversampling : float (default=1)
            Number of super-resolution pixels per camera pixel
        viewport : list or tuple (default=None)
            Field of view to be rendered. If None, all locs are rendered
        blur_method : str (default=None)
            Defines localizations' blur. The string has to be one of
            'gaussian', 'gaussian_iso', 'smooth', 'convolve'. If 'None',
            no blurring is applied.
        min_blur_width : float (default=0)
            Minimum size of blur (pixels)
        ang : tuple (default=None)
            Rotation angles of locs around x, y and z axes. If None,
            locs are not rotated.
    autoscale : boolean (default=False)
        True if optimally adjust contrast
    locs : np.recarray (default=None)
        Locs to be rendered. If None, self.locs is used

    Returns
    -------
    np.array
        8 bit array with 4 channels (rgb and alpha)
    """
    # render each channel one by one
    # get image shape (to avoid rendering unchecked channels)
    (y_min, x_min), (y_max, x_max) = kwargs["viewport"]
    X, Y = (
        int(np.ceil(kwargs["oversampling"] * (x_max - x_min))),
        int(np.ceil(kwargs["oversampling"] * (y_max - y_min))),
    )
    # if single channel is rendered
    if len(locs) == 1:
        renderings = [render.render(_, **kwargs) for _ in locs]
    else:
        renderings = [
            render.render(_, **kwargs) for i, _ in enumerate(locs)
        ]  # renders only channels that are checked in dataset dialog
    # renderings = [render.render(_, **kwargs) for _ in locs]
    # n_locs = sum([_[0] for _ in renderings])
    image = np.array([_[1] for _ in renderings])

    # adjust contrast
    image = scale_contrast(image)

    Y, X = image.shape[1:]
    # array with rgb and alpha channels
    bgra = np.zeros((Y, X, 4), dtype=np.float32)

    colors = get_colors(n_channels=len(locs))

    # color rgb channels and store in bgra
    for color, image in zip(colors, image):
        bgra[:, :, 0] += color[2] * image
        bgra[:, :, 1] += color[1] * image
        bgra[:, :, 2] += color[0] * image

    bgra = np.minimum(bgra, 1)  # minimum value of each pixel is 1
    bgra = to_8bit(bgra)  # convert to 8 bit
    return bgra


def get_colors(n_channels):
    """
    Creates a list with rgb channels for each locs channel.
    Colors go from red to green, blue, pink and red again.

    Parameters
    ----------
    n_channels : int
        Number of locs channels

    Returns
    -------
    list
        Contains tuples with rgb channels
    """

    hues = np.arange(0, 1, 1 / n_channels)
    colors = [colorsys.hsv_to_rgb(_, 1, 1) for _ in hues]
    return colors


def scale_contrast(images):
    """
    Finds optimal contrast for images.

    Parameters
    ----------
    images : list of np.arrays
        Arrays with rendered locs (grayscale)

    Returns
    -------
    list of np.arrays
        Scaled images
    """

    upper = (
        min(
            [
                _.max()
                for _ in images  # if no locs were clustered
                if _.max() != 0  # the maximum value in image is 0.0
            ]
        )
        / 40
    )
    # upper = INITIAL_REL_MAXIMUM * max_

    images = images / upper
    images[~np.isfinite(images)] = 0
    images = np.minimum(images, 1.0)
    images = np.maximum(images, 0.0)
    return images


def to_8bit(image):
    """
    Converts image to 8 bit ready to convert to QImage.

    Parameters
    ----------
    image : np.array
        Image to be converted, with values between 0.0 and 1.0

    Returns
    -------
    np.array
        Image converted to 8 bit
    """
    return np.round(255 * image).astype("uint8")


def get_group_color(locs, n_group_colors):
    """
    Finds group color for each localization in single channel data
    with group info.

    Parameters
    ----------
    locs : np.recarray
        Array with all localizations

    Returns
    -------
    np.array
        Array with int group color index for each loc
    """

    return locs.group.astype(int) % n_group_colors


def get_default_render_kwargs(channel_locs, image_px_size, cam_px_size):
    # Compute the viewport over all channels, ignoring empty channels and
    # non-finite coordinates. Raise ValueError if nothing renderable
    # remains (e.g. a mask that excludes all localizations) so the caller
    # can handle it gracefully instead of producing a NaN viewport.
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    for locs in channel_locs:
        if len(locs) == 0:
            continue
        finite = np.isfinite(locs["x"]) & np.isfinite(locs["y"])
        if not finite.any():
            continue
        x_mins.append(locs["x"][finite].min())
        x_maxs.append(locs["x"][finite].max())
        y_mins.append(locs["y"][finite].min())
        y_maxs.append(locs["y"][finite].max())
    if not x_mins:
        raise ValueError(
            "no localizations with finite coordinates to render"
        )
    x_min, x_max = min(x_mins), max(x_maxs)
    y_min, y_max = min(y_mins), max(y_maxs)

    kwargs = {
        "oversampling": cam_px_size / image_px_size,
        "viewport": [(y_min, x_min), (y_max, x_max)],
        "blur_method": None,
        "min_blur_width": 0,
        "ang": None,
    }
    return kwargs


def plot_scene(
    channel_locs,
    image_px_size,
    cam_px_size,
    fp=None,
    render_kwargs=None,
    x_offset=0,
    y_offset=0,
    title="",
):
    """Plot a scene in the locs
    Args:
        render_kwargs : dict, default None
           optional keys:
            oversampling, viewport, blur_method, min_blur_width, ang,
            n_group_colors, cmap
    """
    if not isinstance(channel_locs, list):
        channel_locs = [channel_locs]

    try:
        kwargs = get_default_render_kwargs(
            channel_locs, image_px_size, cam_px_size
        )
    except ValueError as e:
        logger.error(f"Error plotting locs: {e}")
        nlocs = [len(locs) for locs in channel_locs]
        ngroups = [
            len(np.unique(locs["group"])) if hasattr(locs, "group") else 0
            for locs in channel_locs
        ]
        logger.debug(f"#locs: {nlocs}; #groups{ngroups}")
        fig, ax = plt.subplots()
        if fp is not None:
            fig.savefig(fp)
        return fig, ax
    # overwrite default kwargs with input kwargs
    if render_kwargs is not None:
        for k, v in render_kwargs.items():
            kwargs[k] = v
    x_offset += kwargs["viewport"][0][1] * image_px_size
    y_offset += kwargs["viewport"][0][0] * image_px_size
    logger.debug(f"rendering locs with offset {(x_offset, y_offset)} nm")

    bgra = render_scene(kwargs, channel_locs)

    fig, ax = plt.subplots()
    ax.imshow(
        bgra,
        aspect="equal",
        origin="lower",
        extent=[
            x_offset / 1000,
            (bgra.shape[1] * image_px_size + x_offset) / 1000,
            y_offset / 1000,
            (bgra.shape[0] * image_px_size + y_offset) / 1000,
        ],
    )
    ax.axis("off")
    if fp is not None:
        fig.savefig(fp, bbox_inches="tight", pad_inches=0)
    return fig, ax
