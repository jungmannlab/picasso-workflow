#!/usr/bin/env python
"""Tests for the Doctor tab's cluster-secrets provisioning helpers.

The pure helpers (file-content builders, the remote check command + parser)
and the upload worker's ``_do_upload`` logic are exercised here with a fake
SlurmCommunicator -- no Qt event loop, no SSH, no network. The security
property that matters most is checked explicitly: token VALUES must travel
only inside the written file bodies, never on an ssh/scp command line.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from picasso_workflow.doctor_tab import (  # noqa: E402
    _ClusterSecretsUploadWorker,
    _cluster_check_command,
    _env_file_content,
    _format_cluster_check,
    _parse_cluster_check,
    _secrets_file_content,
    _sh_squote,
)


# -- content builders ---------------------------------------------------------


def test_sh_squote_escapes_single_quotes():
    assert _sh_squote("abc") == "'abc'"
    # a literal single quote becomes the classic '\'' close-escape-reopen
    assert _sh_squote("a'b") == "'a'\\''b'"


def test_secrets_file_content_has_export_lines():
    out = _secrets_file_content("OPTOK", "TESTTOK")
    assert "export CONFLUENCE_TOKEN='OPTOK'" in out
    assert "export TEST_CONFLUENCE_TOKEN='TESTTOK'" in out
    assert out.endswith("\n")


def test_secrets_file_content_omits_absent_tokens():
    # only operational provided -> no TEST line
    out = _secrets_file_content("OPTOK", "")
    assert "export CONFLUENCE_TOKEN='OPTOK'" in out
    assert "TEST_CONFLUENCE_TOKEN" not in out
    # nothing provided -> only the header comments, no export lines
    empty = _secrets_file_content("", "")
    assert "export " not in empty


def test_env_file_content_is_unquoted_dotenv():
    out = _env_file_content("OPTOK")
    # dotenv KEY=value (unquoted; base64url tokens have no spaces/quotes)
    assert "CONFLUENCE_TOKEN=OPTOK" in out
    assert "export " not in out  # .env is NOT a shell script


# -- remote check command + parser --------------------------------------------


def test_cluster_check_command_labels_and_is_value_safe():
    cmd = _cluster_check_command()
    # both files are probed, tagged for parsing
    assert "FILE|env" in cmd
    assert "FILE|secrets" in cmd
    assert "$HOME/.config/picasso_workflow/.env" in cmd
    assert "$HOME/.picasso_secrets" in cmd
    # presence is probed with grep -q (quiet) so no value is ever printed
    assert "grep -q" in cmd
    assert "TEST_CONFLUENCE_TOKEN=" in cmd  # the key, never a value


def test_parse_cluster_check_mixed():
    stdout = (
        "FILE|env|exists|readable=yes|perm=600|op=yes|test=no\n"
        "FILE|secrets|missing\n"
        "some unrelated noise line\n"
    )
    parsed = _parse_cluster_check(stdout)
    assert parsed["env"] == {
        "exists": True,
        "readable": "yes",
        "perm": "600",
        "op": "yes",
        "test": "no",
    }
    assert parsed["secrets"] == {"exists": False}


def test_format_cluster_check_smoke():
    parsed = {
        "env": {"exists": True, "readable": "yes", "perm": "600", "op": "yes"},
        "secrets": {"exists": False},
    }
    html = _format_cluster_check(parsed)
    assert "present" in html
    assert "missing" in html


# -- upload worker (fake communicator; no Qt loop) ----------------------------


class _FakeComm:
    """Records every ssh command and stores every scp'd file body."""

    def __init__(self):
        self.calls = []
        self.files = {}

    def execute_ssh_command(self, command, working_directory=None):
        self.calls.append(command)
        stdout = "/home/alice" if command.startswith("printf") else ""
        return {
            "success": True,
            "stdout": stdout,
            "stderr": "",
            "return_code": 0,
        }

    def put_file(self, content, remote_path, make_executable=False):
        self.files[remote_path] = content
        return {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }


def test_upload_writes_both_files_chmod_600():
    comm = _FakeComm()
    ok, msg = _ClusterSecretsUploadWorker(
        comm, "OPTOK", "TESTTOK"
    )._do_upload()
    assert ok, msg
    env_path = "/home/alice/.config/picasso_workflow/.env"
    secrets_path = "/home/alice/.picasso_secrets"
    assert env_path in comm.files and secrets_path in comm.files
    assert "CONFLUENCE_TOKEN=OPTOK" in comm.files[env_path]
    assert "export CONFLUENCE_TOKEN='OPTOK'" in comm.files[secrets_path]
    assert "export TEST_CONFLUENCE_TOKEN='TESTTOK'" in comm.files[secrets_path]
    # the config dir is created, and both files are tightened to 600
    assert any("mkdir -p" in c for c in comm.calls)
    assert any("chmod 600" in c and ".env" in c for c in comm.calls)
    assert any(
        "chmod 600" in c and ".picasso_secrets" in c for c in comm.calls
    )


def test_upload_never_puts_token_on_a_command_line():
    # The security property: token values live only in the scp'd file bodies,
    # never in an ssh command (which is logged and visible in `ps`).
    comm = _FakeComm()
    _ClusterSecretsUploadWorker(comm, "OPTOK", "TESTTOK")._do_upload()
    assert not any("OPTOK" in c for c in comm.calls)
    assert not any("TESTTOK" in c for c in comm.calls)


def test_upload_aborts_when_home_unresolved():
    class _NoHome(_FakeComm):
        def execute_ssh_command(self, command, working_directory=None):
            self.calls.append(command)
            return {
                "success": True,
                "stdout": "",
                "stderr": "",
                "return_code": 0,
            }

    comm = _NoHome()
    ok, msg = _ClusterSecretsUploadWorker(comm, "OPTOK", "")._do_upload()
    assert not ok
    assert "home" in msg.lower()
    assert comm.files == {}  # nothing written
