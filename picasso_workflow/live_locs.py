#!/usr/bin/env python
"""Reuse lossless live localizations when they are still valid.

The live pass (``LiveAnalysisService``, PycroFlow) localizes frames as they
are acquired. When that pass was **lossless** -- it fitted every frame of the
dataset, nothing was dropped -- its localizations *are* authoritative: the
offline "preprocess" reduction can reuse them wholesale and skip
re-localization entirely (design doc Part III; recommendation §"Authoritative
reduction").

Whether the live locs are reusable is decided by **checksumming the frame
range**: the live descriptor records which frames it covered and a checksum
over that range; a run reuses the live locs only when

1. the descriptor is marked ``lossless`` (every frame fitted), **and**
2. it covers the *whole* dataset's frame range (``[0, n_frames)``), **and**
3. its recorded checksum matches a freshly computed checksum of the same
   frame range of the movie that is about to be analysed.

Any mismatch (a shorter/padded live pass, a re-acquired movie, a lossy live
pass) falls back to re-localizing from the movie -- the safe default.

Data-model note (the tested seam)
---------------------------------
picasso-workflow has no live-locs provenance type of its own yet (that record
is produced upstream by the live service / PycroFlow and will land via WP-4).
So the live descriptor is modelled here as a plain dict
(:class:`LiveLocsDescriptor` documents its keys) that a caller passes into the
``preprocess`` recipe. This module owns the *decision* (``reuse_live_locs``)
and the frame-range checksum (``frame_range_checksum``); wiring a concrete
upstream provenance source into it is a later integration, not a rewrite.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class LiveLocsDescriptor:
    """What the live pass recorded about the locs it produced.

    A thin, self-describing record of a live-localization pass, enough to
    decide reuse. Produced upstream (live service / PycroFlow); modelled as a
    frozen dataclass here so the reuse logic has a typed seam to test against.

    Parameters
    ----------
    locs_path : str
        Where the live localizations were saved (the HDF5 the preprocess
        recipe loads instead of re-localizing).
    frame_start, frame_stop : int
        Half-open frame range the live pass covered: ``[frame_start,
        frame_stop)``.
    checksum : str
        Checksum the live pass computed over its frame range, to be matched
        against a fresh checksum of the candidate movie's same range.
    lossless : bool
        Whether every frame in the range was fitted (nothing dropped). Only a
        lossless pass yields authoritative locs that can be reused as-is.
    """

    locs_path: str
    frame_start: int
    frame_stop: int
    checksum: str
    lossless: bool = False

    @classmethod
    def from_mapping(cls, data: Any) -> "LiveLocsDescriptor | None":
        """Build a descriptor from a plain mapping, or ``None`` if malformed.

        Lets a caller pass the live-locs info as a JSON-ish dict (how it will
        arrive from the upstream service) without constructing the dataclass
        by hand. Returns ``None`` for anything that is not a dict carrying the
        required keys, so a missing/garbled descriptor cleanly means "no
        reuse" rather than raising.
        """
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                locs_path=str(data["locs_path"]),
                frame_start=int(data.get("frame_start", 0)),
                frame_stop=int(data["frame_stop"]),
                checksum=str(data["checksum"]),
                lossless=bool(data.get("lossless", False)),
            )
        except (KeyError, TypeError, ValueError):
            return None


def frame_range_checksum(movie, frame_start: int, frame_stop: int) -> str:
    """Checksum a half-open frame range ``[frame_start, frame_stop)``.

    A content hash of exactly the frames the live pass claims to have
    covered, used to confirm the movie about to be analysed is the same data
    the live locs were computed from. Hashes each frame's raw bytes (plus its
    shape/dtype, so a reshaped-but-equal buffer does not collide) in order.

    Parameters
    ----------
    movie : sequence of ndarray-like
        The movie; ``movie[k]`` is frame ``k`` with ``.shape`` / ``.dtype``
        and ``.tobytes()`` (numpy arrays, picasso movie readers).
    frame_start, frame_stop : int
        The half-open frame range to hash.

    Returns
    -------
    str
        A hex SHA-256 digest of the range.

    Raises
    ------
    IndexError
        If the range exceeds the movie length (the caller is expected to
        clamp to the movie's frame count first; see :func:`reuse_live_locs`).
    """
    h = hashlib.sha256()
    for k in range(frame_start, frame_stop):
        frame = movie[k]
        # Shape + dtype guard a reshaped/retyped buffer from colliding with a
        # genuinely-equal range; then the raw bytes.
        h.update(str(getattr(frame, "shape", "")).encode())
        h.update(str(getattr(frame, "dtype", "")).encode())
        h.update(
            frame.tobytes() if hasattr(frame, "tobytes") else bytes(frame)
        )
    return h.hexdigest()


def _movie_n_frames(movie) -> int:
    """Return a movie's frame count from ``.shape`` or ``len``."""
    shape = getattr(movie, "shape", None)
    if shape:
        return int(shape[0])
    return len(movie)


def reuse_live_locs(
    movie,
    descriptor: Any,
    n_frames: int | None = None,
) -> LiveLocsDescriptor | None:
    """Decide whether the live locs may be reused for this movie.

    Returns the (validated) descriptor when the live localizations are
    authoritative for the whole dataset -- so the recipe can load them and
    skip re-localization -- or ``None`` to re-localize.

    Reuse requires all of:

    1. a well-formed descriptor,
    2. ``lossless`` (every frame fitted upstream),
    3. coverage of the full frame range ``[0, n_frames)``, and
    4. a matching frame-range checksum (the movie is the same data).

    Parameters
    ----------
    movie : sequence of ndarray-like
        The movie about to be analysed.
    descriptor : LiveLocsDescriptor or dict or None
        The live-locs descriptor (dataclass or mapping). ``None`` / malformed
        means no reuse.
    n_frames : int, optional
        The dataset's frame count; inferred from ``movie`` when omitted.

    Returns
    -------
    LiveLocsDescriptor or None
        The validated descriptor if the live locs may be reused, else
        ``None`` (re-localize).
    """
    if not isinstance(descriptor, LiveLocsDescriptor):
        descriptor = LiveLocsDescriptor.from_mapping(descriptor)
    if descriptor is None:
        logger.debug("live-locs reuse: no valid descriptor; re-localizing.")
        return None

    if not descriptor.lossless:
        logger.info(
            "live-locs reuse: live pass was not lossless; re-localizing."
        )
        return None

    if n_frames is None:
        n_frames = _movie_n_frames(movie)

    # The authoritative reduction reuses the live locs only when they cover
    # the WHOLE dataset; a partial live pass is not authoritative.
    if (descriptor.frame_start, descriptor.frame_stop) != (0, n_frames):
        logger.info(
            "live-locs reuse: live range "
            f"[{descriptor.frame_start}, {descriptor.frame_stop}) does not "
            f"cover the full dataset [0, {n_frames}); re-localizing."
        )
        return None

    try:
        actual = frame_range_checksum(movie, 0, n_frames)
    except Exception as e:  # a broken movie read -> fall back to re-localize
        logger.warning(
            f"live-locs reuse: could not checksum frame range ({e!r}); "
            "re-localizing."
        )
        return None

    if actual != descriptor.checksum:
        logger.info(
            "live-locs reuse: frame-range checksum mismatch "
            f"(expected {descriptor.checksum[:12]}…, got {actual[:12]}…); "
            "re-localizing."
        )
        return None

    logger.info(
        "live-locs reuse: lossless live locs are valid for the full frame "
        f"range; reusing {descriptor.locs_path} (no re-localization)."
    )
    return descriptor
