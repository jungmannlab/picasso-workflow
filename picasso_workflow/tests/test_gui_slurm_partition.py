#!/usr/bin/env python
"""GUI test for the Run tab's SLURM Partition dropdown.

Verifies the partition dropdown is populated per cluster from config
(``SlurmPartitions``), preselects the configured default, repopulates when the
cluster changes, and that a chosen partition reaches the generated SLURM
script as a ``#SBATCH --partition=`` directive. Skips gracefully where a Qt
GUI cannot be constructed (no display / PyQt6).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tab test")

from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow import gui  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    try:
        win = gui.Window()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not construct GUI window: {e}")
    yield win
    win.close()


def _configured_partitions(host):
    return [
        str(p) for p in gui.CONFIG.get("SlurmPartitions", {}).get(host, [])
    ]


def test_partition_dropdown_populated_from_config(window):
    """The dropdown lists the partitions configured for the current host."""
    assert hasattr(window, "cluster_partition_combo")
    host = str(window.cluster_host_combo.currentText())
    expected = _configured_partitions(host)
    if not expected:
        pytest.skip(f"No SlurmPartitions configured for host {host!r}")
    items = [
        window.cluster_partition_combo.itemText(i)
        for i in range(window.cluster_partition_combo.count())
    ]
    assert items == expected


def test_partition_dropdown_preselects_configured_default(window):
    """The configured default partition is preselected when it is offered."""
    host = str(window.cluster_host_combo.currentText())
    default = str(gui.CONFIG.get("SlurmDefault", {}).get("partition", ""))
    if default not in _configured_partitions(host):
        pytest.skip("Configured default partition not offered for this host")
    assert window.cluster_partition_combo.currentText() == default


def test_partition_dropdown_clears_for_host_without_partitions(window):
    """A host with no configured partitions leaves the field blank, so no
    --partition is emitted."""
    window._populate_cluster_partitions("nodeXX")
    assert window.cluster_partition_combo.currentText() == ""
    assert window.cluster_partition_combo.count() == 0


def test_slurm_script_includes_partition_directive():
    """A chosen partition surfaces as a #SBATCH --partition directive; an
    empty one is omitted."""
    comm = gui.SlurmCommunicator("host", "user")
    with_partition = comm.create_slurm_script(
        "job", ["echo hi"], slurm_options={"partition": "p.hpcl8"}
    )
    assert "#SBATCH --partition=p.hpcl8" in with_partition

    without = comm.create_slurm_script("job", ["echo hi"], slurm_options={})
    assert "--partition" not in without


def test_slurm_script_propagates_workflow_exit_code():
    """The batch script captures the srun step's status and exits with it, so a
    killed/failed workflow is reported as FAILED (not COMPLETED)."""
    comm = gui.SlurmCommunicator("host", "user")
    commands = comm.assemble_slurm_commands("hpcl8XXX", scriptname="wf.py")
    # srun immediately followed by capturing its exit status
    srun_i = next(i for i, c in enumerate(commands) if c.startswith("srun "))
    assert commands[srun_i + 1] == "PW_RC=$?"

    script = comm.create_slurm_script("job", commands)
    # the script exits with the captured code as its final statement
    lines = [ln.strip() for ln in script.splitlines() if ln.strip()]
    assert lines[-1] == "exit ${PW_RC:-0}"
    # srun (and its capture) come before the exit
    assert "srun python wf.py" in script

    # other callers, which never set PW_RC, still exit 0 (${PW_RC:-0})
    plain = comm.create_slurm_script("job", ["echo hi"])
    assert plain.strip().splitlines()[-1].strip() == "exit ${PW_RC:-0}"


def test_slurm_commands_load_configured_modules():
    """Configured environment modules (e.g. cuda for GPU fitting) are emitted
    as `module load` lines, and ~/.local shadowing is disabled."""
    host = next(iter(gui.CONFIG.get("SlurmLoginNodes", {})), None)
    modules = (
        gui.CONFIG.get("ClusterEnvironment", {})
        .get(host, {})
        .get("Modules", [])
    )
    if not modules:
        pytest.skip(f"No Modules configured for host {host!r}")
    comm = gui.SlurmCommunicator("host", "user")
    commands = comm.assemble_slurm_commands(host)
    for module_name in modules:
        assert f"module load {module_name}" in commands
    assert "export PYTHONNOUSERSITE=1" in commands
    # faulthandler dumps the Python->C stack on a native crash (e.g. a LAPACK
    # SIGSEGV in a fit/CRLB call), so segfaults are pinpointed, not silent.
    assert "export PYTHONFAULTHANDLER=1" in commands
    # module loads must precede the python launch that relies on them
    launch = next(i for i, c in enumerate(commands) if "python" in c)
    for module_name in modules:
        assert commands.index(f"module load {module_name}") < launch


def test_modules_override_and_empty():
    """An explicit modules list overrides config; an empty list loads
    nothing."""
    comm = gui.SlurmCommunicator("host", "user")
    override = comm.assemble_slurm_commands("hpcl8XXX", modules=["cuda/12.4"])
    assert "module load cuda/12.4" in override
    assert "module load cuda/13.0" not in override  # config default replaced

    none = comm.assemble_slurm_commands("hpcl8XXX", modules=[])
    assert not any(c.startswith("module load") for c in none)


def test_modules_field_prefilled_from_config(window):
    """The Modules field is prefilled (space-separated) from the current
    cluster's configured modules."""
    assert hasattr(window, "cluster_modules_edit")
    host = str(window.cluster_host_combo.currentText())
    expected = " ".join(window._configured_modules(host))
    assert window.cluster_modules_edit.text() == expected


def test_modules_field_preserves_manual_edit_on_cluster_change(window):
    """A manual edit to the Modules field survives a cluster change; an
    untouched field follows the new cluster's config."""
    window.cluster_modules_edit.setText("cuda/12.4 gcc/12")
    window._populate_cluster_modules("nodeXX")
    assert window.cluster_modules_edit.text() == "cuda/12.4 gcc/12"
