"""Unit tests for the headless in-app self-update engine (no network/pip)."""

import sys

from picasso_workflow import self_update


def test_is_bundled_install(tmp_path):
    assert not self_update.is_bundled_install(prefix=str(tmp_path))
    (tmp_path / self_update.BUNDLED_MARKER).write_text("")
    assert self_update.is_bundled_install(prefix=str(tmp_path))


def test_wheel_asset_url_found():
    release = {
        "assets": [
            {
                "name": "picasso-workflow-0.8.7-win.exe",
                "browser_download_url": "https://x/app.exe",
            },
            {
                "name": "picasso_workflow-0.8.7-py3-none-any.whl",
                "browser_download_url": "https://x/pw.whl",
            },
        ]
    }
    assert self_update.wheel_asset_url(release) == "https://x/pw.whl"


def test_wheel_asset_url_absent():
    assert self_update.wheel_asset_url({"assets": []}) is None
    assert self_update.wheel_asset_url({}) is None
    assert (
        self_update.wheel_asset_url(
            {
                "assets": [
                    {
                        "name": "app.exe",
                        "browser_download_url": "https://x/app.exe",
                    }
                ]
            }
        )
        is None
    )


def test_wheel_asset_url_filters_to_package_and_pins_version():
    release = {
        "assets": [
            {
                "name": "someother-1.0-py3-none-any.whl",
                "browser_download_url": "https://x/other.whl",
            },
            {
                "name": "picasso_workflow-0.8.7-py3-none-any.whl",
                "browser_download_url": "https://x/pw87.whl",
            },
        ]
    }
    # skips the non-package wheel
    assert self_update.wheel_asset_url(release) == "https://x/pw87.whl"
    # version pin: matching version returned
    assert (
        self_update.wheel_asset_url(release, version="0.8.7")
        == "https://x/pw87.whl"
    )
    # version pin: a mismatched version is refused (don't install a stale wheel)
    assert self_update.wheel_asset_url(release, version="0.9.0") is None


def test_update_log_path():
    assert self_update.update_log_path(prefix="/opt/pw").endswith("update.log")


def test_install_dir_writable(tmp_path):
    import os

    # a writable dir probes True
    assert self_update.install_dir_writable(target=str(tmp_path)) is True
    # a non-existent dir probes False (mkstemp raises)
    assert (
        self_update.install_dir_writable(target=str(tmp_path / "nope"))
        is False
    )
    # a read-only dir probes False (the all-users / ProgramData case). Skip
    # when running as root, where mode bits don't block writes.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        assert self_update.install_dir_writable(target=str(ro)) is False
    finally:
        os.chmod(ro, 0o700)  # let pytest clean up


def test_read_update_outcome(tmp_path):
    # no log -> None
    assert self_update.read_update_outcome(prefix=str(tmp_path)) is None
    # OK log -> ("OK", detail) and the log is consumed (shown once)
    log = tmp_path / "update.log"
    log.write_text("OK\nSuccessfully installed picasso_workflow-0.8.11\n")
    status, detail = self_update.read_update_outcome(prefix=str(tmp_path))
    assert status == "OK"
    assert "0.8.11" in detail
    assert not log.exists()  # deleted after reading
    # FAILED log -> status + reason surfaced
    log.write_text("FAILED\nERROR: Access is denied: ...site-packages\n")
    status, detail = self_update.read_update_outcome(prefix=str(tmp_path))
    assert status == "FAILED"
    assert "Access is denied" in detail


def test_updater_source_runs_argv_wait_and_logs(tmp_path):
    # Actually EXECUTE the detached-updater source as a subprocess, with a
    # definitely-dead parent pid (so the wait exits at once) and a bogus local
    # wheel (so pip fails fast, no network). Exercises the argv indexing under
    # `python -c`, _alive(), the wait->install path, and the log write -- a
    # reorder of sys.argv[1..3] or a broken _alive would be caught here.
    import subprocess

    log = tmp_path / "update.log"
    dead_pid = "2147483647"  # not a live pid -> _alive() False immediately
    bogus_wheel = str(tmp_path / "no_such_package.whl")  # local -> pip, no net
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            self_update._UPDATER_SOURCE,
            bogus_wheel,
            dead_pid,
            str(log),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0  # the helper never raises; it logs the outcome
    assert log.exists()
    # dead parent -> it did NOT skip; it ran pip (which failed on the bogus
    # wheel), so the log records FAILED (never SKIPPED here).
    head = log.read_text().splitlines()[0]
    assert head == "FAILED"


def test_spawn_detached_updater_builds_detached_pip_command():
    captured = {}

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return object()

    log = self_update.spawn_detached_updater(
        "https://x/pw.whl",
        parent_pid=4321,
        python="/env/python",
        log_path="/p/update.log",
        popen=fake_popen,
    )
    cmd = captured["cmd"]
    assert log == "/p/update.log"
    assert cmd[0] == "/env/python" and cmd[1] == "-c"
    # the helper source + its three args (wheel, parent pid, log path)
    assert "pip" in cmd[2] and "_alive" in cmd[2]
    assert cmd[3] == "https://x/pw.whl"
    assert cmd[4] == "4321"
    assert cmd[5] == "/p/update.log"
    # detached: own session/process-group, never a child that dies with the GUI
    assert captured["kw"].get("close_fds") is True
    assert "creationflags" in captured["kw"] or captured["kw"].get(
        "start_new_session"
    )


def test_updater_source_waits_for_parent_then_pips():
    src = self_update._UPDATER_SOURCE
    # waits on the parent pid, then runs pip upgrade of the wheel
    assert "_alive(parent_pid)" in src
    assert "pip" in src and "install" in src and "--upgrade" in src
    # imports no picasso_workflow (so it never holds the DLLs it must replace)
    assert "picasso_workflow" not in src
