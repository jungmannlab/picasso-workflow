#!/usr/bin/env python
"""Tests for picasso_workflow._env.configure_numba_cache.

The helper redirects numba's on-disk cache to a per-user writable directory so
a shared/read-only install does not fail when numba tries to write its cache
next to the package. Loaded directly from its file so importing it does not
trigger the heavy ``picasso_workflow/__init__.py`` (which pulls picasso/numba).
"""

import importlib.util
import os
import pathlib

_ENV_PATH = pathlib.Path(__file__).resolve().parents[1] / "_env.py"


def _load_env():
    spec = importlib.util.spec_from_file_location(
        "pwf_env_under_test", _ENV_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sets_cache_dir_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows expanduser
    env = _load_env()

    result = env.configure_numba_cache()

    expected = str(tmp_path / ".picasso_workflow" / "numba_cache")
    assert result == expected
    assert os.environ["NUMBA_CACHE_DIR"] == expected
    assert os.path.isdir(expected)


def test_respects_existing_value(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_CACHE_DIR", "/some/explicit/dir")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    env = _load_env()

    result = env.configure_numba_cache()

    assert result is None
    assert os.environ["NUMBA_CACHE_DIR"] == "/some/explicit/dir"
    # did not create its own cache tree
    assert not (tmp_path / ".picasso_workflow").exists()


def test_home_not_writable_leaves_env_untouched(tmp_path, monkeypatch):
    # Point HOME at a regular file so makedirs under it raises OSError.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(blocker))
    monkeypatch.setenv("USERPROFILE", str(blocker))
    env = _load_env()

    result = env.configure_numba_cache()

    assert result is None
    assert "NUMBA_CACHE_DIR" not in os.environ


def test_unresolvable_home_leaves_env_untouched(monkeypatch):
    # When no home can be resolved, os.path.expanduser("~") returns "~"
    # unchanged; the helper must not create a literal "~" dir / set a relative
    # NUMBA_CACHE_DIR. (A bare delenv("HOME") would still resolve via the passwd
    # database, so patch expanduser directly.)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr(os.path, "expanduser", lambda p: "~")
    env = _load_env()

    result = env.configure_numba_cache()

    assert result is None
    assert "NUMBA_CACHE_DIR" not in os.environ
