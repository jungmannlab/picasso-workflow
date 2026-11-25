"""Unit tests for the headless SSH-to-SLURM onboarding engine.

The real SSH transport (SlurmCommunicator) and the filesystem are mocked, so
these run in the unit tier with no network, no cluster and no display.
"""

import os

import pytest

from picasso_workflow import ssh_setup


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeComm:
    """Stand-in for SlurmCommunicator recording its ctor args."""

    def __init__(self, result=None, raises=None):
        self.result = result or {}
        self.raises = raises

    def __call__(
        self, hostname, username, port=22, ssh_key_path=None, timeout=30
    ):
        self.hostname = hostname
        self.username = username
        self.port = port
        self.ssh_key_path = ssh_key_path
        self.timeout = timeout
        return self

    def execute_ssh_command(self, command, working_directory=None):
        if self.raises is not None:
            raise self.raises
        return self.result


def _ok_result():
    return {
        "stdout": "picasso-ssh-ok\n",
        "stderr": "",
        "return_code": 0,
        "success": True,
    }


def _fail_result(stderr):
    return {
        "stdout": "",
        "stderr": stderr,
        "return_code": 255,
        "success": False,
    }


# --------------------------------------------------------------------------- #
# test_ssh
# --------------------------------------------------------------------------- #
def test_ssh_success():
    comm = _FakeComm(result=_ok_result())
    v = ssh_setup.test_ssh("login01", "alice", comm_factory=comm)
    assert v.ok and v.category == ssh_setup.CAT_OK
    # ctor received the connection params
    assert comm.hostname == "login01" and comm.username == "alice"


def test_ssh_requires_host_and_user():
    v = ssh_setup.test_ssh("", "alice", comm_factory=_FakeComm(_ok_result()))
    assert not v.ok and v.category == ssh_setup.CAT_ERROR


def test_ssh_missing_key_file(tmp_path):
    missing = str(tmp_path / "nope")
    v = ssh_setup.test_ssh(
        "login01",
        "alice",
        ssh_key_path=missing,
        comm_factory=_FakeComm(_ok_result()),
    )
    assert not v.ok and v.category == ssh_setup.CAT_NO_KEY


@pytest.mark.parametrize(
    "stderr,category",
    [
        ("alice@login01: Permission denied (publickey).", ssh_setup.CAT_AUTH),
        (
            "ssh: Could not resolve hostname login01: Name or service not "
            "known",
            ssh_setup.CAT_UNKNOWN_HOST,
        ),
        (
            "ssh: connect to host login01 port 22: Connection timed out",
            ssh_setup.CAT_TIMEOUT,
        ),
        (
            "Permissions 0644 for 'id_ed25519' are too open.",
            ssh_setup.CAT_KEY_PERMS,
        ),
        ("some unexpected failure", ssh_setup.CAT_ERROR),
    ],
)
def test_ssh_classifies_failures(stderr, category):
    comm = _FakeComm(result=_fail_result(stderr))
    v = ssh_setup.test_ssh("login01", "alice", comm_factory=comm)
    assert not v.ok and v.category == category
    assert v.detail == stderr


def test_ssh_missing_ssh_binary():
    comm = _FakeComm(raises=FileNotFoundError("ssh"))
    v = ssh_setup.test_ssh("login01", "alice", comm_factory=comm)
    assert not v.ok and v.category == ssh_setup.CAT_ERROR
    assert "ssh" in v.message.lower()


# --------------------------------------------------------------------------- #
# key discovery / generation / commands
# --------------------------------------------------------------------------- #
def test_discover_ssh_key_prefers_rsa(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "id_ed25519").write_text("x")
    assert ssh_setup.discover_ssh_key(home=tmp_path).endswith("id_ed25519")
    (ssh / "id_rsa").write_text("x")
    assert ssh_setup.discover_ssh_key(home=tmp_path).endswith("id_rsa")


def test_discover_ssh_key_none(tmp_path):
    assert ssh_setup.discover_ssh_key(home=tmp_path) is None


def test_generate_ssh_key_invokes_keygen(tmp_path):
    calls = {}

    def fake_runner(cmd, **kw):
        calls["cmd"] = cmd
        return type("P", (), {"returncode": 0, "stderr": ""})()

    key = str(tmp_path / ".ssh" / "id_ed25519")
    out = ssh_setup.generate_ssh_key(key_path=key, runner=fake_runner)
    assert out == key
    assert calls["cmd"][:4] == ["ssh-keygen", "-t", "ed25519", "-N"]
    assert key in calls["cmd"]


def test_generate_ssh_key_refuses_overwrite(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    key = ssh / "id_ed25519"
    key.write_text("existing")
    with pytest.raises(FileExistsError):
        ssh_setup.generate_ssh_key(
            key_path=str(key), runner=lambda *a, **k: None
        )


def test_generate_ssh_key_raises_on_failure(tmp_path):
    def bad_runner(cmd, **kw):
        return type("P", (), {"returncode": 1, "stderr": "boom"})()

    with pytest.raises(RuntimeError):
        ssh_setup.generate_ssh_key(
            key_path=str(tmp_path / "k"), runner=bad_runner
        )


def test_ssh_copy_id_command():
    cmd = ssh_setup.ssh_copy_id_command(
        "login01", "alice", port=2222, key_path="/h/.ssh/id_ed25519"
    )
    assert cmd == (
        "ssh-copy-id -p 2222 -i /h/.ssh/id_ed25519.pub alice@login01"
    )
    # default port omits -p
    assert ssh_setup.ssh_copy_id_command("login01", "alice") == (
        "ssh-copy-id alice@login01"
    )


def test_manual_install_command_pipes_pub():
    cmd = ssh_setup.manual_install_command(
        "login01", "alice", key_path="/h/.ssh/id_ed25519"
    )
    assert cmd.startswith('cat "/h/.ssh/id_ed25519.pub" | ssh alice@login01')
    assert "authorized_keys" in cmd


def test_resolve_username_expands_dollar_user(monkeypatch):
    monkeypatch.setattr(ssh_setup.getpass, "getuser", lambda: "realuser")
    assert ssh_setup.resolve_username("$USER") == "realuser"
    assert ssh_setup.resolve_username("") == "realuser"
    assert ssh_setup.resolve_username("   ") == "realuser"
    assert ssh_setup.resolve_username(None) == "realuser"
    assert ssh_setup.resolve_username("alice") == "alice"
    assert ssh_setup.resolve_username("  alice  ") == "alice"


def test_ssh_copy_id_command_string_port():
    # A string port must not break the "!= 22" comparison, and a non-numeric
    # port degrades to the default (no -p) instead of raising.
    assert ssh_setup.ssh_copy_id_command("h", "u", port="2222") == (
        "ssh-copy-id -p 2222 u@h"
    )
    assert ssh_setup.ssh_copy_id_command("h", "u", port="bad") == (
        "ssh-copy-id u@h"
    )


# --------------------------------------------------------------------------- #
# state detection
# --------------------------------------------------------------------------- #
def test_detect_state_dict_usernames(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "id_rsa").write_text("x")
    cfg = {
        "SlurmLoginNodes": {"clusterA": "login01"},
        "LoginNodeUserNames": {"clusterA": "alice"},
    }
    st = ssh_setup.detect_state(config=cfg, home=tmp_path)
    assert st.hosts == {"clusterA": "login01"}
    assert st.usernames == {"clusterA": "alice"}
    assert st.has_key and st.key_path.endswith("id_rsa")


def test_detect_state_string_username(tmp_path):
    cfg = {
        "SlurmLoginNodes": {"clusterA": "login01"},
        "LoginNodeUserNames": "$USER",
    }
    st = ssh_setup.detect_state(config=cfg, home=tmp_path)
    assert st.usernames == {"clusterA": "$USER"}
    assert not st.has_key


# --------------------------------------------------------------------------- #
# config persistence
# --------------------------------------------------------------------------- #
def test_write_user_config_new_file(tmp_path):
    import yaml

    cfg = tmp_path / "config.yaml"
    ssh_setup.write_user_config(
        {"SlurmLoginNodes": {"clusterA": "login01"}}, config_path=str(cfg)
    )
    data = yaml.safe_load(cfg.read_text())
    assert data["SlurmLoginNodes"]["clusterA"] == "login01"
    assert not (tmp_path / "config.yaml.bak").exists()


def test_write_user_config_merges_and_backs_up(tmp_path):
    import yaml

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "SlurmLoginNodes": {"old": "oldhost"},
                "Confluence": {"url": "keep"},
            }
        )
    )
    ssh_setup.write_user_config(
        {
            "SlurmLoginNodes": {"clusterA": "login01"},
            "LoginNodeUserNames": {"clusterA": "alice"},
        },
        config_path=str(cfg),
    )
    data = yaml.safe_load(cfg.read_text())
    # merged: old host kept, new host added, unrelated section untouched
    assert data["SlurmLoginNodes"] == {"old": "oldhost", "clusterA": "login01"}
    assert data["LoginNodeUserNames"] == {"clusterA": "alice"}
    assert data["Confluence"] == {"url": "keep"}
    # backup written with the pre-merge content
    bak = yaml.safe_load((tmp_path / "config.yaml.bak").read_text())
    assert "clusterA" not in bak["SlurmLoginNodes"]


def test_persist_cluster_ssh_writes_keys_and_reloads(tmp_path, monkeypatch):
    import yaml

    reloaded = {}
    monkeypatch.setattr(
        ssh_setup, "reload", lambda: reloaded.setdefault("called", True)
    )
    cfg = tmp_path / "config.yaml"
    ssh_setup.persist_cluster_ssh(
        "login01", "login01", "alice", config_path=str(cfg)
    )
    data = yaml.safe_load(cfg.read_text())
    assert data["SlurmLoginNodes"]["login01"] == "login01"
    assert data["LoginNodeUserNames"]["login01"] == "alice"
    assert reloaded.get("called") is True


def test_write_user_config_atomic_no_tmp_left(tmp_path):
    cfg = tmp_path / "config.yaml"
    ssh_setup.write_user_config(
        {"SlurmLoginNodes": {"a": "b"}}, config_path=str(cfg)
    )
    assert not os.path.exists(str(cfg) + ".tmp")
