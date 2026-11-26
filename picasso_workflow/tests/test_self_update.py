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


def test_pip_upgrade_command_uses_this_env_and_only_if_needed():
    cmd = self_update.pip_upgrade_command("https://x/pw.whl")
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "pip", "install"]
    assert "--upgrade" in cmd
    i = cmd.index("--upgrade-strategy")
    assert cmd[i + 1] == "only-if-needed"
    assert cmd[-1] == "https://x/pw.whl"


def _proc(returncode=0, stderr="", stdout=""):
    return type(
        "P", (), {"returncode": returncode, "stderr": stderr, "stdout": stdout}
    )()


def test_run_update_success_tells_user_to_restart():
    calls = {}

    def runner(cmd, **kw):
        calls["cmd"] = cmd
        return _proc(0)

    ok, msg = self_update.run_update("https://x/pw.whl", runner=runner)
    assert ok
    assert "reopen" in msg.lower() or "restart" in msg.lower()
    assert calls["cmd"][-1] == "https://x/pw.whl"


def test_run_update_no_wheel():
    ok, msg = self_update.run_update(None, runner=lambda *a, **k: _proc(0))
    assert not ok and "no update wheel" in msg.lower()


def test_run_update_locked_file_message():
    def runner(cmd, **kw):
        return _proc(
            1,
            stderr="ERROR: [WinError 32] being used by another "
            "process: 'Qt6Core.dll'",
        )

    ok, msg = self_update.run_update("https://x/pw.whl", runner=runner)
    assert not ok
    assert "in use" in msg.lower() and "close" in msg.lower()


def test_run_update_generic_failure_surfaces_error():
    def runner(cmd, **kw):
        return _proc(1, stderr="ERROR: Could not find a version")

    ok, msg = self_update.run_update("https://x/pw.whl", runner=runner)
    assert not ok and "could not find a version" in msg.lower()
