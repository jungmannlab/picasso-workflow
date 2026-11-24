#!/usr/bin/env python
"""Unit tests for the GUI SLURM path's Apptainer containerization.

The GUI's "Start Workflow on Cluster" path always runs the workflow INSIDE an
Apptainer image on a shared mount, so the cluster runs the same picasso-workflow
version the scientist drives the GUI with. These tests exercise the Qt-free
pieces in :mod:`picasso_workflow.slurm_submit`:

* :func:`default_image_basename` -- version -> ``.sif`` filename mapping (and
  its deliberate ``None`` fall-backs for ``unknown``/dev versions);
* :meth:`SlurmCommunicator.assemble_slurm_commands` -- the ``apptainer exec``
  wrap (bind mounts, ``--nv``, module-load pin) vs. the unchanged host-conda
  ``srun python`` fall-back.

No Qt, no SSH, no network -- just the command-string assembly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from picasso_workflow.slurm_submit import (
    DEFAULT_APPTAINER_MODULE,
    SlurmCommunicator,
    apptainer_config,
    apptainer_exec_prefix,
    default_image_basename,
    resolve_cluster_image_path,
)


# --------------------------------------------------------------------------- #
# hardening (PR #96 follow-up): empty/None handling, module fallback, cd quote
# --------------------------------------------------------------------------- #
def test_apptainer_config_drops_blank_bind_entries():
    cfg = {
        "ClusterApptainer": {
            "image_dir": "/sw",
            "bind_paths": ["/a", None, "", "  ", "/b "],
        }
    }
    _dir, _mod, binds = apptainer_config("h", config=cfg)
    # null/blank entries dropped, survivors stripped
    assert binds == ["/a", "/b"]


def test_apptainer_exec_prefix_skips_empty_binds():
    out = apptainer_exec_prefix("/sw/x.sif", ["/a", "", "/b"])
    assert out == "apptainer exec --bind /a --bind /b /sw/x.sif"


def test_empty_image_path_is_host_conda_not_containerized():
    # "" is the host-conda sentinel just like None -- must NOT emit `apptainer
    # exec '' ...`.
    cmds = SlurmCommunicator("h", "u").assemble_slurm_commands(
        "hpcl8XXX", image_path=""
    )
    assert "srun python start_workflow.py" in cmds
    assert not any("apptainer" in c for c in cmds)


def test_containerized_module_falls_back_to_default():
    # image_path given but no apptainer_module -> still load the pinned default
    # (else `apptainer` isn't on PATH on a module-based HPC).
    cmds = SlurmCommunicator("h", "u").assemble_slurm_commands(
        "hpcl8XXX", image_path="/sw/x.sif"
    )
    assert f"module load {DEFAULT_APPTAINER_MODULE}" in cmds


def test_create_slurm_script_quotes_working_dir():
    s = SlurmCommunicator("h", "u").create_slurm_script(
        "job", ["echo hi"], working_directory="/res/my run"
    )
    assert "cd '/res/my run'" in s


# --------------------------------------------------------------------------- #
# resolve_cluster_image_path
# --------------------------------------------------------------------------- #
def test_resolve_cluster_image_path():
    # bare name -> joined onto image_dir
    assert resolve_cluster_image_path("x.sif", "/sw/pw") == "/sw/pw/x.sif"
    # absolute path -> used as-is (image_dir ignored)
    assert (
        resolve_cluster_image_path("/other/x.sif", "/sw/pw") == "/other/x.sif"
    )
    # empty selection, or no image_dir to join onto -> unchanged
    assert resolve_cluster_image_path("", "/sw/pw") == ""
    assert resolve_cluster_image_path("x.sif", "") == "x.sif"


# --------------------------------------------------------------------------- #
# config_template.yaml -- ClusterApptainer section
# --------------------------------------------------------------------------- #
def test_config_template_has_clusterapptainer_section():
    """The bundled default ships the ClusterApptainer keys the GUI reads."""
    template = Path(__file__).resolve().parents[1] / "config_template.yaml"
    cfg = yaml.safe_load(template.read_text())
    section = cfg["ClusterApptainer"]
    assert "image_dir" in section
    assert "module" in section
    assert isinstance(section.get("bind_paths"), list)


def test_config_template_admin_console_hidden_by_default():
    """The admin-console menu is restricted: hidden unless opted in."""
    template = Path(__file__).resolve().parents[1] / "config_template.yaml"
    cfg = yaml.safe_load(template.read_text())
    assert cfg["AdminConsole"]["enabled"] is False


def _comm():
    return SlurmCommunicator("login-node", "user")


# --------------------------------------------------------------------------- #
# default_image_basename
# --------------------------------------------------------------------------- #
def test_default_image_basename_release_version():
    assert default_image_basename("0.9.0") == "picasso-workflow-0.9.0.sif"


def test_default_image_basename_unknown_returns_none():
    # A non-git install reports __version__ == "unknown"; no reliable name.
    assert default_image_basename("unknown") is None
    assert default_image_basename("") is None
    assert default_image_basename(None) is None


def test_default_image_basename_dev_version_returns_none():
    # setuptools-scm dev versions map to a develop-<sha> CI image, which the
    # version string cannot reconstruct -> leave the pick to the dropdown.
    assert default_image_basename("0.9.1.dev3+g1a2b3c4") is None
    assert default_image_basename("0.9.1.dev3") is None


# --------------------------------------------------------------------------- #
# apptainer_exec_prefix -- shared by GUI path and cluster-submit worker
# --------------------------------------------------------------------------- #
def test_apptainer_exec_prefix_basic():
    assert (
        apptainer_exec_prefix("/sw/x.sif", ["/scratch/paint"])
        == "apptainer exec --bind /scratch/paint /sw/x.sif"
    )


def test_apptainer_exec_prefix_nv_and_quoting():
    out = apptainer_exec_prefix(
        "/sw/x.sif", ["/scratch/my data", "/fs/t"], use_nv=True
    )
    assert out == (
        "apptainer exec --nv --bind '/scratch/my data' "
        "--bind /fs/t /sw/x.sif"
    )


def test_apptainer_exec_prefix_no_binds():
    assert apptainer_exec_prefix("/sw/x.sif") == "apptainer exec /sw/x.sif"


def test_apptainer_exec_prefix_env():
    out = apptainer_exec_prefix(
        "/sw/x.sif",
        ["/scratch/run"],
        env={"NUMBA_CACHE_DIR": "/scratch/run/.numba"},
    )
    # --env comes before --bind and the image
    assert out == (
        "apptainer exec --env NUMBA_CACHE_DIR=/scratch/run/.numba "
        "--bind /scratch/run /sw/x.sif"
    )


# --------------------------------------------------------------------------- #
# assemble_slurm_commands -- containerized
# --------------------------------------------------------------------------- #
def test_containerized_run_wraps_in_apptainer_exec():
    cmds = _comm().assemble_slurm_commands(
        "hpcl8XXX",
        scriptname="start_workflow.py",
        image_path="/sw/pw/picasso-workflow-0.9.0.sif",
        bind_paths=["/scratch/paint"],
        apptainer_module="apptainer/1.4.1",
    )
    # apptainer module is loaded (so `apptainer` resolves)...
    assert "module load apptainer/1.4.1" in cmds
    # ...and the workflow runs inside the image, with the bind mount.
    run = next(c for c in cmds if c.startswith("srun apptainer exec"))
    assert "--bind /scratch/paint" in run
    assert "/sw/pw/picasso-workflow-0.9.0.sif" in run
    assert run.endswith("python start_workflow.py")
    # exit status is still captured right after the run.
    assert cmds[cmds.index(run) + 1] == "PW_RC=$?"
    # inherited-into-container env knobs stay.
    assert "export PYTHONNOUSERSITE=1" in cmds
    assert "export PYTHONFAULTHANDLER=1" in cmds


def test_containerized_run_redirects_numba_cache():
    # Inside the read-only image numba cannot write its default cache next to
    # the package; the run must redirect it (and matplotlib's config) under the
    # user's home, set BEFORE numba is imported (inherited into the container).
    cmds = _comm().assemble_slurm_commands(
        "hpcl8XXX",
        image_path="/sw/pw/x.sif",
        bind_paths=["/scratch/paint"],
    )
    assert (
        'export NUMBA_CACHE_DIR="$HOME/.picasso_workflow/numba_cache"' in cmds
    )
    assert 'export MPLCONFIGDIR="$HOME/.picasso_workflow/mpl"' in cmds
    # the dirs are created before use
    assert any(c.startswith("mkdir -p") and "numba_cache" in c for c in cmds)
    # and the export precedes the apptainer run (so numba sees it)
    run_i = next(
        i for i, c in enumerate(cmds) if c.startswith("srun apptainer exec")
    )
    nb_i = cmds.index(
        'export NUMBA_CACHE_DIR="$HOME/.picasso_workflow/numba_cache"'
    )
    assert nb_i < run_i


def test_host_conda_run_does_not_redirect_numba_cache():
    # Host-conda runs use the user's own (writable) env; no redirect needed.
    cmds = _comm().assemble_slurm_commands("hpcl8XXX", image_path=None)
    assert not any("NUMBA_CACHE_DIR" in c for c in cmds)
    assert not any("MPLCONFIGDIR" in c for c in cmds)


def test_containerized_run_omits_host_conda_wiring():
    cmds = _comm().assemble_slurm_commands(
        "hpcl8XXX",
        image_path="/sw/pw/x.sif",
        bind_paths=["/scratch/paint"],
        apptainer_module="apptainer/1.4.1",
        modules=["cuda/13.0"],
    )
    joined = "\n".join(cmds)
    # No host conda-env exports (the interpreter comes from the image)...
    assert "CONDA_DEFAULT_ENV" not in joined
    assert "CONDA_PREFIX" not in joined
    # ...and the host CUDA module is NOT loaded (CUDA lives in the image).
    assert "module load cuda/13.0" not in joined
    # no plain `srun python` -- it must be the apptainer form.
    assert not any(c == "srun python start_workflow.py" for c in cmds)


def test_nv_flag_only_when_requested():
    with_nv = _comm().assemble_slurm_commands(
        "hpcl8XXX",
        image_path="/sw/pw/x.sif",
        bind_paths=["/scratch/paint"],
        use_nv=True,
    )
    assert any("apptainer exec --nv" in c for c in with_nv)

    without = _comm().assemble_slurm_commands(
        "hpcl8XXX",
        image_path="/sw/pw/x.sif",
        bind_paths=["/scratch/paint"],
        use_nv=False,
    )
    assert not any("--nv" in c for c in without)


def test_bind_paths_are_shell_quoted():
    # Lab data folders routinely contain spaces; the apptainer exec line is a
    # real shell command, so bind/image paths must be quoted.
    cmds = _comm().assemble_slurm_commands(
        "hpcl8XXX",
        image_path="/sw/pw/x.sif",
        bind_paths=["/scratch/my data"],
    )
    run = next(c for c in cmds if c.startswith("srun apptainer exec"))
    assert "--bind '/scratch/my data'" in run


# --------------------------------------------------------------------------- #
# assemble_slurm_commands -- host-conda fall-back (image_path=None)
# --------------------------------------------------------------------------- #
def test_non_containerized_run_is_plain_srun_python():
    cmds = _comm().assemble_slurm_commands(
        "hpcl8XXX", scriptname="start_workflow.py"
    )
    assert "srun python start_workflow.py" in cmds
    assert not any("apptainer" in c for c in cmds)
    # exit status capture unchanged.
    srun_i = next(i for i, c in enumerate(cmds) if c.startswith("srun "))
    assert cmds[srun_i + 1] == "PW_RC=$?"


# --------------------------------------------------------------------------- #
# secrets provisioning -- assemble_slurm_commands sources ~/.picasso_secrets
# on the node before the run (long-standing behaviour; locked in here because
# the containerized path relies on it -- Apptainer then forwards the exported
# CONFLUENCE_TOKEN into the image via its default host-env passthrough).
# --------------------------------------------------------------------------- #
SECRETS_SOURCE = "[ -f ~/.picasso_secrets ] && source ~/.picasso_secrets"


def test_secrets_sourced_before_srun_containerized():
    # The operational token file is sourced on the node BEFORE the run so
    # CONFLUENCE_TOKEN is in the env; Apptainer forwards it into the image.
    cmds = _comm().assemble_slurm_commands(
        "hpcl8XXX",
        image_path="/sw/pw/x.sif",
        bind_paths=["/scratch/paint"],
    )
    assert SECRETS_SOURCE in cmds
    # sourced before the workflow runs (so the export is in effect).
    srun_i = next(i for i, c in enumerate(cmds) if c.startswith("srun "))
    assert cmds.index(SECRETS_SOURCE) < srun_i


def test_secrets_sourced_before_srun_host_conda():
    # Same provisioning on the non-containerized (host-conda) path.
    cmds = _comm().assemble_slurm_commands(
        "hpcl8XXX", scriptname="start_workflow.py"
    )
    assert SECRETS_SOURCE in cmds
    srun_i = next(i for i, c in enumerate(cmds) if c.startswith("srun "))
    assert cmds.index(SECRETS_SOURCE) < srun_i


def test_secrets_source_never_embeds_the_token():
    # The token is only *sourced* at run time -- never written into the
    # script -- so a generated sbatch file stays free of the secret.
    cmds = _comm().assemble_slurm_commands(
        "hpcl8XXX", image_path="/sw/pw/x.sif"
    )
    joined = "\n".join(cmds)
    assert "CONFLUENCE_TOKEN=" not in joined
    assert "CONFLUENCE_BEARER=" not in joined
