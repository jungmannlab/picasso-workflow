"""Zeiss ``.czi`` -> picasso ``.raw`` conversion (extracted from picasso_outpost).

`keep` -- niche import glue. picasso reads ``.czi`` natively via ``io.load_czi``
(needs the ``[formats]`` extra: aicsimageio).
"""

from picasso_workflow._picasso_api import io


def convert_zeiss_file(filepath_czi, filepath_raw, info=None):
    """Convert a Zeiss ``.czi`` file into a picasso-readable ``.raw`` file.

    Parameters
    ----------
    filepath_czi : str
        Filepath of the ``.czi`` file to load.
    filepath_raw : str
        Filepath of the ``.raw`` file to write.
    info : dict, optional
        Metadata to make the raw file picasso-readable. If None, dummy values
        are entered. Necessary keys: ``'Byte Order'``, ``'Camera'``,
        ``'Micro-Manager Metadata'``.
    """
    # picasso reads Zeiss .czi natively: io.load_czi returns (movie, [info]),
    # where the movie is array-like and reduces to a (frames, y, x) array.
    movie, _czi_info = io.load_czi(filepath_czi)
    data = movie[:].squeeze()

    if info is None:
        info = {"Byte Order": "<", "Camera": "FusionBT"}
        info["File"] = filepath_raw
        info["Height"] = data.shape[-2]
        info["Width"] = data.shape[-1]
        info["Frames"] = data.shape[0]
        info["Data Type"] = data.dtype.name
        info["Micro-Manager Metadata"] = {
            "FusionBT-ReadoutMode": 1,
            "Filter": 561,
        }

    # save_raw writes both the ``.raw`` movie and its ``.yaml`` sidecar.
    io.save_raw(filepath_raw, data, [info])
