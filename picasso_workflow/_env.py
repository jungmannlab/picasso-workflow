"""Process-environment setup that must run before heavy third-party imports.

Kept dependency-free (standard library only) and called at the very top of
``picasso_workflow/__init__.py`` -- before ``picasso`` (and, through it,
``numba``) is imported -- so the configuration is in place when numba reads it.
"""

from __future__ import annotations

import os


def configure_numba_cache() -> str | None:
    """Point numba's on-disk cache at a per-user writable directory.

    picasso compiles ``@numba.njit(cache=True)`` functions whose compiled
    cache numba writes *next to the package* in ``site-packages`` by default.
    On a shared or read-only install that location is not writable by the user
    running the app:

    * an all-users Windows install (prefix under ``%ALLUSERSPROFILE%``, e.g.
      ``C:\\ProgramData\\picasso-workflow``) that a non-admin user launches, or
    * the read-only Apptainer image used on the cluster,

    so the cache write fails and picasso's import / first call errors out.
    Redirect the cache to ``~/.picasso_workflow/numba_cache`` (the same per-user
    tree the logs already use), which every user can write.

    Only sets ``NUMBA_CACHE_DIR`` when it is not already set, so an explicit
    override wins (e.g. a cluster run passing it via ``apptainer --env``), and
    only when the directory can actually be created -- otherwise numba keeps its
    default behaviour rather than being pointed at an unusable path.

    Must run before numba is first imported.

    Returns
    -------
    str or None
        The directory configured, or ``None`` if the environment was left
        untouched (already set, or the directory could not be created).
    """
    if os.environ.get("NUMBA_CACHE_DIR"):
        return None
    home = os.path.expanduser("~")
    if home == "~":
        # No home directory resolvable (no HOME / USERPROFILE and no passwd
        # entry): expanduser left "~" unchanged. Leave numba's default rather
        # than creating a literal "~" directory in the current working dir and
        # pointing numba at a relative path.
        return None
    cache_dir = os.path.join(home, ".picasso_workflow", "numba_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        # Home not writable (rare) -- leave numba's default rather than
        # pointing it at a directory that does not exist.
        return None
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
    return cache_dir
