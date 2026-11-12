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
    SlurmCommunicator,
    apptainer_exec_prefix,
    default_image_basename,
)


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
