#!/usr/bin/env python
"""Qt-free SLURM/SSH submit transport for picasso-workflow.

``SlurmCommunicator`` is the reusable SSH + sbatch-over-SSH machinery the GUI
(``gui.py``) has always used to submit cluster jobs. It lives here, apart from
``gui.py``, so that **non-GUI** callers -- notably the WP-9 cluster-submit
worker (:mod:`picasso_workflow.cluster_submit`) -- can reuse the exact same
transport without importing PyQt (``gui.py`` imports PyQt6 at module level).

The class is pure ``subprocess`` / ``scp`` / stdlib: no Qt, no registry, no
network at import. ``gui.py`` re-exports it (``from picasso_workflow.slurm_submit
import SlurmCommunicator``) so its own call sites and the existing GUI tests are
unchanged.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import subprocess
import tempfile

from loguru import logger

from picasso_workflow import CONFIG

#: Apptainer env module pinned to the LOWEST version on the target node group
#: (register C10: hpcl8 max 1.4.1; hpcl10/hpcl20 up to 1.5.2 -- pin to the
#: lowest so the same submit template runs on every node group). Overridable
#: via config ``ClusterApptainer.module`` (GUI path) or
#: ``ClusterSubmitConfig.apptainer_module`` (cluster-submit worker). Single
#: source of the pin; ``cluster_submit`` re-imports it.
DEFAULT_APPTAINER_MODULE = "apptainer/1.4.1"


def resolve_pw_version():
    """The installed ``picasso_workflow`` version, or ``"unknown"``.

    Prefers the dist metadata (present even in a non-git wheel install, where
    ``_version.py`` / ``__version__`` may be absent) and falls back to the
    importable ``__version__``. Single source so the Run tab's image auto-match
    and the admin console report the SAME version (the match highlighting
    depends on it).
    """
    try:
        import importlib.metadata

        return importlib.metadata.version("picasso_workflow")
    except Exception:
        try:
            from picasso_workflow import __version__

            return __version__
        except Exception:
            return "unknown"


def default_image_basename(version):
    """Return the shared-mount ``.sif`` filename matching ``version``.

    The GUI auto-selects the Apptainer image whose filename encodes its own
    installed ``picasso_workflow`` version, so the cluster runs the SAME version
    the scientist is driving the GUI with. The naming mirrors the CI release
    build (``.github/workflows/build-apptainer.yml``):
    ``picasso-workflow-<version>.sif``.

    Returns ``None`` when no reliable name can be formed -- i.e. an ``"unknown"``
    version (a non-git install, see ``picasso_workflow.__init__``) or a
    setuptools-scm dev version (``...dev...``/``+g<sha>``), whose CI image is
    instead named ``picasso-workflow-develop-<sha>.sif`` and cannot be
    reconstructed from the version string. In those cases the caller falls back
    to the Run tab's image dropdown rather than guessing a filename.
    """
    if not version or version == "unknown":
        return None
    # setuptools-scm dev versions ("0.9.1.dev3+g1a2b3c4") don't map to a
    # release .sif; the matching CI artefact is the develop-<sha> image, which
    # this string can't reconstruct. Leave the selection to the dropdown.
    if "dev" in version or "+" in version:
        return None
    return f"picasso-workflow-{version}.sif"


def _merged_apptainer_config(host_cluster, config=None):
    """The ``ClusterApptainer`` block with an optional per-host override merged.

    The override is a sub-dict keyed as in ``SlurmLoginNodes``; only the keys it
    sets replace the globals. Shared by :func:`apptainer_config` and
    :func:`apptainer_test_results_dir` so the merge lives in one place.
    """
    if config is None:
        config = CONFIG
    base = dict(config.get("ClusterApptainer", {}) or {})
    override = base.get(str(host_cluster))
    if isinstance(override, dict):
        base.update(override)
    return base


def apptainer_config(host_cluster, config=None):
    """Resolved Apptainer settings for ``host_cluster``.

    Returns ``(image_dir, module, bind_paths)`` from the per-host-merged
    ``ClusterApptainer`` block. Shared by the GUI Run tab and the admin console
    so both resolve the shared mount the same way. ``config`` defaults to the
    package ``CONFIG``.
    """
    base = _merged_apptainer_config(host_cluster, config)
    image_dir = str(base.get("image_dir", "") or "")
    module = base.get("module") or DEFAULT_APPTAINER_MODULE
    # Drop null/blank entries so a stray `-` or empty line under
    # ClusterApptainer.bind_paths doesn't become `--bind None` / `--bind ''`
    # (which apptainer rejects).
    bind_paths = [
        s
        for b in (base.get("bind_paths") or [])
        if b is not None and (s := str(b).strip())
    ]
    return image_dir, module, bind_paths


def apptainer_test_results_dir(host_cluster, config=None):
    """The cluster scratch dir for admin test-run artefacts (or ``""``).

    ``ClusterApptainer.test_results_dir`` (per-host-merged like
    :func:`apptainer_config`). A per-run subdirectory is created under it so
    runs never intermingle; no login-node repo is needed.
    """
    base = _merged_apptainer_config(host_cluster, config)
    return str(base.get("test_results_dir", "") or "")


def resolve_cluster_image_path(selection, image_dir):
    """Resolve a user-typed image selection to a cluster path.

    An absolute POSIX path is used as-is; a bare filename is joined onto
    ``image_dir``; an empty selection (or no ``image_dir`` to join onto) is
    returned unchanged. Shared by the Run tab and the admin Test-runs tab so a
    typed basename resolves the same way in both.
    """
    selection = (selection or "").strip()
    if posixpath.isabs(selection):
        return selection
    if selection and image_dir:
        return posixpath.join(image_dir, selection)
    return selection


def apptainer_exec_prefix(image_path, bind_paths=None, use_nv=False, env=None):
    """Build the ``apptainer exec [--nv] [--env ...] --bind ... <image>`` prefix.

    Shared by the GUI submit path (:meth:`assemble_slurm_commands`), the
    cluster-submit worker (:func:`picasso_workflow.cluster_submit.
    build_sbatch_script`), and the admin test-run orchestration so the
    container invocation is assembled once. The caller appends the in-container
    command (``python ...``).

    Paths are ``shlex``-quoted because this is a real shell command and lab
    data folders routinely contain spaces (e.g. ``channel 0.hdf5``); an
    unquoted space would word-split the argv. ``use_nv`` adds ``--nv`` to
    expose the host NVIDIA driver (the CUDA toolkit lives in the image).
    ``env`` is an optional mapping of ``NAME -> value`` passed through as
    ``--env NAME=value`` (e.g. writable ``NUMBA_CACHE_DIR``/``MPLCONFIGDIR`` so
    picasso can import inside the read-only image, or ``PW_TEST_DATA_DIR``).
    """
    parts = ["apptainer exec"]
    if use_nv:
        parts.append("--nv")
    for name, value in (env or {}).items():
        parts.append(f"--env {shlex.quote(f'{name}={value}')}")
    parts += [f"--bind {shlex.quote(b)}" for b in (bind_paths or []) if b]
    parts.append(shlex.quote(image_path))
    return " ".join(parts)


class SlurmCommunicator:
    """Communication interface for SLURM job scheduling and SSH command
    execution.

    This class provides methods to interact with SLURM job schedulers through
    SSH, create SLURM job scripts, submit jobs, and monitor their status. It's
    designed to work with remote compute clusters running SLURM workload
    manager.

    Attributes
    ----------
    hostname : str
        SSH hostname or IP address
    username : str
        SSH username for authentication
    port : int
        SSH port number (default: 22)
    ssh_key_path : str
        Path to SSH private key file
    timeout : int
        SSH connection timeout in seconds
    """

    def __init__(
        self, hostname, username, port=22, ssh_key_path=None, timeout=30
    ):
        """Initialize SLURM communicator with SSH connection parameters.

        Parameters
        ----------
        hostname : str
            SSH hostname or IP address
        username : str
            SSH username for authentication
        port : int, optional
            SSH port number. Defaults to 22.
        ssh_key_path : str, optional
            Path to SSH private key file.
            If None, uses default SSH authentication.
        timeout : int, optional
            SSH connection timeout in seconds. Defaults to 30.
        """
        self.hostname = hostname
        self.username = username
        self.port = port
        self.ssh_key_path = ssh_key_path
        self.timeout = timeout

        logger.info(
            f"Initialized SlurmCommunicator for {username}@{hostname}:{port}"
        )

    def execute_ssh_command(self, command, working_directory=None):
        """Execute a command via SSH on the remote host.

        Parameters
        ----------
        command : str
            Shell command to execute
        working_directory : str, optional
            Directory to execute command in

        Returns
        -------
        dict: Dictionary containing:
            - stdout (str): Standard output
            - stderr (str): Standard error
            - return_code (int): Command exit code
            - success (bool): True if return_code == 0

        Raises
        ------
        subprocess.CalledProcessError: If SSH connection fails
        FileNotFoundError: If SSH executable not found
        """
        ssh_cmd = [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "ConnectTimeout={}".format(self.timeout),
            "-o",
            "StrictHostKeyChecking=no",
        ]

        if self.ssh_key_path:
            ssh_cmd.extend(["-i", self.ssh_key_path])

        # Construct full command
        if working_directory:
            remote_command = f"cd {working_directory} && {command}"
        else:
            remote_command = command

        ssh_cmd.extend([f"{self.username}@{self.hostname}", remote_command])

        logger.debug(f"Executing SSH command: {' '.join(ssh_cmd)}")

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
                * 2,  # Allow extra time for command execution
            )

            response = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
            }

            if not response["success"]:
                logger.warning(
                    "SSH command failed with return code "
                    + f"{result.returncode}: {result.stderr}"
                )
            else:
                logger.debug("SSH command completed successfully")

            return response

        except subprocess.TimeoutExpired:
            logger.error(
                f"SSH command timed out after {self.timeout * 2} seconds"
            )
            return {
                "stdout": "",
                "stderr": (
                    f"Command timed out after {self.timeout * 2} seconds"
                ),
                "return_code": -1,
                "success": False,
            }
        except Exception as e:
            logger.error(f"SSH command execution failed: {str(e)}")
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "success": False,
            }

    def test_connection(self):
        """Test SSH connection to the remote host.

        Returns
        -------
        bool: True if connection successful, False otherwise
        """
        result = self.execute_ssh_command('echo "Connection test successful"')
        return result["success"]

    def assemble_slurm_commands(
        self,
        host_cluster,
        scriptname="start_workflow.py",
        modules=None,
        faulthandler=True,
        cpu_only=False,
        image_path=None,
        bind_paths=None,
        apptainer_module=None,
        use_nv=False,
    ):
        """Assembles picasso-workflow specific commands for running a batch
        job on a SLURM cluster.

        ``modules`` overrides the environment modules to ``module load`` (a
        list of names). None falls back to the cluster's configured
        ``Modules``; an empty list loads nothing.

        ``faulthandler`` exports ``PYTHONFAULTHANDLER=1`` so a native crash
        (e.g. a LAPACK/CUDA fault in a fit/CRLB call) prints a Python->C
        traceback instead of dying silently. ``cpu_only`` exports
        ``CUDA_VISIBLE_DEVICES=""`` to hide every GPU, forcing picasso's CPU
        fit/CRLB kernels -- a debug switch to isolate a GPU-side crash from the
        CPU path.

        When ``image_path`` is given the workflow runs INSIDE that Apptainer
        image (``apptainer exec``) rather than the host conda env, so the
        cluster runs the exact picasso-workflow version the image was built
        with -- the GUI path always sets it. ``apptainer_module`` is the env
        module to ``module load`` so ``apptainer`` resolves (mirrors
        :func:`picasso_workflow.cluster_submit.build_sbatch_script`);
        ``bind_paths`` are host roots bound into the container (it only sees
        what is bound in); ``use_nv`` adds ``--nv`` to expose the host NVIDIA
        driver for GPU fitting (the CUDA toolkit lives in the image). With
        ``image_path`` left ``None`` (or empty) the command stays the
        host-conda ``srun python`` for any non-GUI caller.
        """
        # Treat empty-string like None: an empty image_path must NOT emit
        # `apptainer exec '' ...` (the sentinel for "host conda" is falsy).
        containerized = bool(image_path)
        # A host with no ClusterEnvironment entry (e.g. an unconfigured
        # cluster, or a minimal config with no bundled config.yaml) degrades
        # gracefully: every lookup below is a .get() with a default, so no
        # path/module exports are emitted rather than crashing.
        cluster_env = (
            CONFIG.get("ClusterEnvironment", {}).get(host_cluster) or {}
        )

        commands = []

        commands.append("source ~/.bashrc")

        # Confluence credentials live in a dedicated secrets file rather than
        # ~/.bashrc, whose non-interactive guard would `return` before
        # reaching any exports inside a batch job. The file has no such
        # guard, so it always runs (and is a no-op if absent).
        secrets_file = "~/.picasso_secrets"
        commands.append(f"[ -f {secrets_file} ] && source {secrets_file}")

        commands.append("source /etc/profile.d/modules.sh")
        # if use_pw_module:
        #     commands.append(f"module load {pw_module}")
        # else:
        #     commands.append(f"module load {anaconda_module}")

        # if not use_pw_module:
        #     commands.append(f"conda activate {conda_env}")

        # Load environment modules. Containerized runs load ONLY the Apptainer
        # module (CUDA et al. live inside the image, exposed via --nv), so the
        # host cuda/conda modules are skipped. Host-conda runs load the Run
        # tab's Modules field (or config ClusterEnvironment.<host>.Modules when
        # not overridden): GPU fitting (e.g. spline-mle-gpu) needs a CUDA
        # toolkit that provides libNVVM; `module load cuda/<ver>` also sets
        # CUDA_HOME so numba finds it. Harmless on CPU-only runs.
        if containerized:
            # Always load an Apptainer module in container mode so `apptainer`
            # resolves on a module-based HPC; fall back to the pinned default
            # when the caller didn't pass one (the GUI always does).
            module = apptainer_module or DEFAULT_APPTAINER_MODULE
            commands.append(f"module load {shlex.quote(module)}")
        else:
            if modules is None:
                modules = cluster_env.get("Modules", []) or []
            for module_name in modules:
                commands.append(f"module load {module_name}")

        # Ignore ~/.local/lib site-packages: a stray user-site install (e.g.
        # numba) otherwise shadows the conda env's package and can break CUDA
        # discovery. Keep the job pinned to the env resolved below.
        commands.append("export PYTHONNOUSERSITE=1")

        # Dump the Python -> C traceback on fatal signals (e.g. a LAPACK
        # SIGSEGV inside a native fit/CRLB call, which leaves no Python
        # exception). Inherited by multiprocessing workers, so a crash in a
        # pool worker is pinpointed too. Negligible overhead.
        if faulthandler:
            commands.append("export PYTHONFAULTHANDLER=1")

        # Debug switch: hide every GPU so picasso's CUDA probe
        # (cuda.is_available()) returns False and the fit/CRLB run on the
        # (fully-guarded) CPU kernels. Isolates a GPU-side crash -- e.g. a
        # spline CRLB kernel fault that dies as a hard SIGSEGV instead of a
        # catchable CudaAPIError -- from the CPU path. Note picasso runs the
        # spline CRLB on the GPU whenever one is visible, even for the CPU
        # 'spline-mle' fit method, so this is the only lever that keeps it off.
        if cpu_only:
            commands.append('export CUDA_VISIBLE_DEVICES=""')

        # Host-conda env wiring (PATH/LD_LIBRARY_PATH/PYTHONPATH + CONDA_*).
        # Skipped for containerized runs: the interpreter and its sys.path come
        # from the image, so these host-env tweaks would be meaningless (or
        # point CONDA_* at a non-existent host env).
        if not containerized:
            bin_path = cluster_env.get("BinPath", None)
            if bin_path:
                commands.append(f"export PATH={bin_path}:$PATH")
            lib_path = cluster_env.get("LibraryPath", None)
            if lib_path:
                commands.append(
                    f"export LD_LIBRARY_PATH={lib_path}:$LD_LIBRARY_PATH"
                )
            python_path = cluster_env.get("PythonPath", None)
            if python_path:
                commands.append(f"export PYTHONPATH={python_path}:$PYTHONPATH")
            conda_env = cluster_env.get("CondaEnv", None)
            if conda_env:
                commands.append(f"export CONDA_DEFAULT_ENV={conda_env}")
            conda_prefix = cluster_env.get("CondaPrefix", None)
            if conda_prefix:
                commands.append(f"export CONDA_PREFIX={conda_prefix}")

        # Capture the workflow's exit status. srun propagates the step's
        # failure (e.g. 143 = 128 + SIGTERM when a rank is killed / an MPI
        # teardown cancels the step). create_slurm_script exits the batch
        # script with this, so a failed/cancelled run is reported as FAILED
        # rather than COMPLETED (a bare `srun ...` followed by an `echo` would
        # otherwise make the batch script exit 0 regardless).
        #
        # Containerized: run the script INSIDE the image. The exported env above
        # (PYTHONNOUSERSITE/PYTHONFAULTHANDLER/CUDA_VISIBLE_DEVICES) is inherited
        # into the container -- Apptainer passes the host env through by default.
        if containerized:
            prefix = apptainer_exec_prefix(image_path, bind_paths, use_nv)
            commands.append(f"srun {prefix} python {scriptname}")
        else:
            commands.append(f"srun python {scriptname}")
        commands.append("PW_RC=$?")

        return commands

    def create_slurm_script(
        self,
        job_name,
        commands,
        slurm_options=None,
        output_file=None,
        error_file=None,
        working_directory=None,
    ):
        """Create a SLURM job script with specified parameters.

        Parameters
        ----------
        job_name : str
            Name for the SLURM job
        commands : list or str
            List of shell commands or single command string
        slurm_options : dict, optional
            SLURM directive options. Keys should be SLURM parameter names
            (without --), values should be parameter values.
            Common options:
                  - partition: Queue/partition name
                  - nodes: Number of nodes
                  - ntasks: Number of tasks
                  - cpus-per-task: CPUs per task
                  - mem: Memory per node
                  - time: Time limit (HH:MM:SS)
                  - mail-type: Email notification type
                  - mail-user: Email address
        output_file : str, optional
            Path for stdout redirection
        error_file : str, optional
            Path for stderr redirection
        working_directory : str, optional
            Working directory for the job

        Returns
        -------
        str: Complete SLURM script content
        """
        script_lines = ["#!/bin/bash"]

        # Add SLURM directives
        script_lines.append(f"#SBATCH --job-name={job_name}")

        if output_file:
            script_lines.append(f"#SBATCH --output={output_file}")
        else:
            script_lines.append(f"#SBATCH --output={job_name}_%j.out")

        if error_file:
            script_lines.append(f"#SBATCH --error={error_file}")
        else:
            script_lines.append(f"#SBATCH --error={job_name}_%j.err")

        # Add additional SLURM options
        if slurm_options:
            for option, value in slurm_options.items():
                if value is not None:
                    script_lines.append(f"#SBATCH --{option}={value}")

        script_lines.append("")  # Empty line after SLURM directives

        # Add working directory change if specified. Quote it: the containerized
        # run `cd`s here and then launches a RELATIVE start_workflow.py, so a
        # results folder containing a space must not word-split the `cd`.
        if working_directory:
            script_lines.append(f"cd {shlex.quote(working_directory)}")
            script_lines.append("")

        # Add timestamp and job info
        script_lines.extend(
            [
                'echo "Job started at: $(date)"',
                'echo "Running on node: $(hostname)"',
                'echo "Job ID: $SLURM_JOB_ID"',
                'echo "Current working directory: $(pwd)"',
                'echo ""',
            ]
        )

        # Add user commands
        if isinstance(commands, str):
            commands = [commands]

        for cmd in commands:
            script_lines.append(cmd)

        # Add job completion timestamp, then exit with the workflow's status so
        # SLURM reports the job's real outcome. PW_RC is set by
        # assemble_slurm_commands after the srun step; ${PW_RC:-0} keeps other
        # callers (which never set it) exiting 0 as before.
        script_lines.extend(
            [
                "",
                'echo ""',
                'echo "Job completed at: $(date) (exit ${PW_RC:-0})"',
                "exit ${PW_RC:-0}",
            ]
        )

        return "\n".join(script_lines)

    def write_slurm_script(self, script_content, folder, local=True):
        """Write SLURM script content to a file on the remote host.

        Parameters
        ----------
        script_content : str
            Complete SLURM script content
        folder : str
            Path where to save the script on remote host
        local : bool
            whether the folder is available on the local system

        Returns
        -------
        dict: Result of the write operation (see execute_ssh_command)
        """
        if local:
            filepath = os.path.join(folder, "run_workflow_slurm.sh")
            # write with UNIX style newlines ('\n') instead of DOS ('\r\n')
            with open(filepath, "w", newline="\n") as f:
                f.write(script_content)
            return filepath

        # otherwise, copy to remote host
        # Create a temporary local file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as tmp_file:
            tmp_file.write(script_content)
            tmp_file_path = tmp_file.name

        try:
            # Copy file to remote host using scp (shared flags via _scp).
            result = self._scp(tmp_file_path, folder)

            if result.returncode == 0:
                # Make script executable
                chmod_result = self.execute_ssh_command(f"chmod +x {folder}")
                if chmod_result["success"]:
                    logger.info(
                        f"SLURM script written successfully to {folder}"
                    )
                    return {
                        "stdout": f"Script written to {folder}",
                        "stderr": "",
                        "return_code": 0,
                        "success": True,
                    }
                else:
                    return chmod_result
            else:
                logger.error(f"Failed to copy SLURM script: {result.stderr}")
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "success": False,
                }

        except Exception as e:
            logger.error(f"Error writing SLURM script: {str(e)}")
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "success": False,
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass

    def put_file(self, content, remote_path, make_executable=False):
        """Write ``content`` to an exact ``remote_path`` over scp.

        A general staging primitive (unlike :meth:`write_slurm_script`, which
        targets a fixed filename in a folder): the admin test-run orchestration
        stages several named files -- per-tier sbatch scripts, ``summarize.py``,
        ``jobs.tsv`` -- into a cluster run directory. Returns the usual
        result dict (``success``/``stdout``/``stderr``/``return_code``).
        """
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, newline="\n"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = self._scp(tmp_path, remote_path)
            if result.returncode != 0:
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "success": False,
                }
            if make_executable:
                return self.execute_ssh_command(
                    f"chmod +x {shlex.quote(remote_path)}"
                )
            return {
                "stdout": f"wrote {remote_path}",
                "stderr": "",
                "return_code": 0,
                "success": True,
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _scp(self, local_path, remote_dest):
        """Run ``scp local_path user@host:remote_dest``; return the result.

        The scp flags (port, timeout, StrictHostKeyChecking, optional ``-i``
        key) live here so :meth:`put_file` and :meth:`write_slurm_script` stay
        in lockstep. Returns the ``subprocess.CompletedProcess``.
        """
        scp_cmd = [
            "scp",
            "-P",
            str(self.port),
            "-o",
            "ConnectTimeout={}".format(self.timeout),
            "-o",
            "StrictHostKeyChecking=no",
        ]
        if self.ssh_key_path:
            scp_cmd.extend(["-i", self.ssh_key_path])
        scp_cmd.extend(
            [local_path, f"{self.username}@{self.hostname}:{remote_dest}"]
        )
        return subprocess.run(
            scp_cmd, capture_output=True, text=True, timeout=self.timeout
        )

    def submit_job(self, script_path, dest_machine, additional_options=None):
        """Submit a SLURM job using sbatch.

        Parameters
        ----------
        script_path : str
            Path to the SLURM script on remote host
        additional_options : list, optional
            Additional sbatch options

        Returns
        -------
        dict: Result containing job submission information:
            - job_id (int or None): SLURM job ID if successful
            - stdout (str): sbatch output
            - stderr (str): sbatch errors
            - success (bool): True if job submitted successfully
        """
        # check script path. if it is local, convert to its location on
        # the remote host
        from picasso_workflow.metaworkflow import PathParser

        pp = PathParser()
        script_path = pp.convert_path(script_path, dest_machine)

        sbatch_cmd = "sbatch"

        if additional_options:
            sbatch_cmd += " " + " ".join(additional_options)

        # Quote the script path for the remote shell: cluster/lab paths contain
        # spaces, which would otherwise split `sbatch <half-a-path>` into a
        # bogus argv (the rest of the class already quotes remote paths).
        sbatch_cmd += f" {shlex.quote(script_path)}"

        logger.info(f"Submitting job via SSH: {sbatch_cmd}")
        result = self.execute_ssh_command(sbatch_cmd)

        # Parse job ID from output
        job_id = None
        if result["success"] and result["stdout"]:
            # sbatch typically outputs: "Submitted batch job 12345"
            match = re.search(r"Submitted batch job (\d+)", result["stdout"])
            if match:
                job_id = int(match.group(1))
                logger.info(f"Job submitted successfully with ID: {job_id}")
            else:
                logger.warning("Job submitted but could not parse job ID")

        return {
            "job_id": job_id,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "success": result["success"],
        }

    def get_job_status(self, job_id):
        """Get the status of a SLURM job.

        Parameters
        ----------
        job_id : int
            SLURM job ID

        Returns
        -------
        dict: Job status information:
            - status (str):
                Job status (PENDING, RUNNING, COMPLETED, FAILED, etc.)
            - details (dict): Additional job information
            - success (bool): True if status query successful
        """
        squeue_cmd = f'squeue --job={job_id} --format="%T %R %S %E" --noheader'
        result = self.execute_ssh_command(squeue_cmd)

        if result["success"] and result["stdout"].strip():
            # Parse squeue output
            fields = result["stdout"].strip().split()
            if len(fields) >= 1:
                status = fields[0]
                reason = fields[1] if len(fields) > 1 else ""
                start_time = fields[2] if len(fields) > 2 else ""
                end_time = fields[3] if len(fields) > 3 else ""

                return {
                    "status": status,
                    "details": {
                        "reason": reason,
                        "start_time": start_time,
                        "end_time": end_time,
                    },
                    "success": True,
                }

        # If squeue doesn't show the job, it has left the queue: query sacct
        # for the terminal state plus the diagnostics that explain a failure
        # (exit code, peak memory for OOM, elapsed time). squeue drops a job
        # the moment it finishes, so this fallback is the only way to learn
        # whether it COMPLETED, FAILED, TIMEOUT or was OUT_OF_MEMORY.
        sacct_cmd = (
            f"sacct --job={job_id} "
            "--format=State,ExitCode,MaxRSS,Elapsed --noheader --parsable2"
        )
        sacct_result = self.execute_ssh_command(sacct_cmd)

        if sacct_result["success"] and sacct_result["stdout"].strip():
            # first line is the top-level job step; take the widest state.
            line = sacct_result["stdout"].strip().split("\n")[0]
            fields = line.split("|")
            status = fields[0] if fields else "UNKNOWN"
            details = {}
            if len(fields) > 1 and fields[1]:
                details["exit_code"] = fields[1]
            if len(fields) > 2 and fields[2]:
                details["max_rss"] = fields[2]
            if len(fields) > 3 and fields[3]:
                details["elapsed"] = fields[3]
            return {"status": status, "details": details, "success": True}

        return {
            "status": "UNKNOWN",
            "details": {"error": result["stderr"]},
            "success": False,
        }

    def fetch_progress(self, remote_folder):
        """Fetch the most recent ``progress.json`` under ``remote_folder``.

        A run writes ``progress.json`` into a timestamped subfolder whose name
        is only known at runtime, so we locate the newest one by mtime over
        SSH and cat it. This reuses the same SSH transport as the job-status
        queries -- no cluster-side daemon is needed.

        Parameters
        ----------
        remote_folder : str
            Remote path of the run's results folder.

        Returns
        -------
        dict or None
            The parsed progress state, or None if not found/unreadable.
        """
        if not remote_folder:
            return None
        folder = shlex.quote(remote_folder)
        # newest progress.json by modification time, path only
        find_cmd = (
            f"find {folder} -name progress.json -printf '%T@ %p\\n' "
            "2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-"
        )
        res = self.execute_ssh_command(find_cmd)
        path = (res.get("stdout") or "").strip()
        if not path:
            return None
        cat = self.execute_ssh_command(f"cat {shlex.quote(path)}")
        if not cat.get("success"):
            return None
        try:
            return json.loads(cat["stdout"])
        except (ValueError, KeyError):
            return None

    def fetch_all_progress(self, remote_folder):
        """Fetch *every* ``progress.json`` under ``remote_folder`` in one call.

        An aggregation run has one progress file per stage (top-level
        aggregation, each single dataset, the aggregation stage). Fetching
        them individually would be one slow SSH round-trip each, so this cats
        them all in a single command, separated by a marker, and splits the
        result client-side.

        Parameters
        ----------
        remote_folder : str
            Remote path of the run's results folder.

        Returns
        -------
        list of dict
            Parsed progress states (empty if none / unreachable).
        """
        if not remote_folder:
            return []
        folder = shlex.quote(remote_folder)
        sep = "===PWF-PROGRESS-SEP==="
        # Rank 0 writes ``progress.json``; worker ranks write
        # ``progress.rankN.json`` (see progress.default_sinks). Match both, or a
        # multi-rank aggregation run reports only rank 0's share of the datasets.
        # ``-exec`` passes each match as an argument, so paths containing spaces
        # or newlines survive (a ``for f in $(find ...)`` word-splits them and
        # loses the file, blanking the monitor for a genuinely running job).
        cmd = (
            f"find {folder} \\( -name progress.json -o "
            f"-name 'progress.rank*.json' \\) "
            f'-exec sh -c \'echo "{sep}"; cat "$1"\' _ {{}} \\; 2>/dev/null'
        )
        res = self.execute_ssh_command(cmd)
        out = res.get("stdout") or ""
        states = []
        for chunk in out.split(sep):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                states.append(json.loads(chunk))
            except ValueError:
                continue
        return states

    def write_abort_flag(self, remote_folder):
        """Drop an ``abort.flag`` in every run folder under ``remote_folder``.

        The running workflow checks this file between modules and inside long
        picasso calls, so this requests a graceful stop (complementary to
        ``scancel``, which kills hard).

        Parameters
        ----------
        remote_folder : str
            Remote path of the run's results folder.

        Returns
        -------
        dict
            The SSH command result.
        """
        folder = shlex.quote(remote_folder)
        # Place a flag next to every progress.json (single + aggregation runs).
        # ``-exec`` with ``dirname "$1"`` handles paths with spaces/newlines; a
        # ``for d in $(find ...)`` word-splits them, so a graceful abort would
        # silently miss those folders and never reach the running workflow.
        cmd = (
            f"find {folder} -name progress.json -exec sh -c "
            '\'touch "$(dirname "$1")/abort.flag"\' _ {} \\; '
            f"2>/dev/null; touch {folder}/abort.flag"
        )
        return self.execute_ssh_command(cmd)

    def cancel_job(self, job_id):
        """Cancel a SLURM job.

        Parameters
        ----------
        job_id : int
            SLURM job ID to cancel

        Returns
        -------
        dict: Result of the cancellation (see execute_ssh_command)
        """
        scancel_cmd = f"scancel {job_id}"
        result = self.execute_ssh_command(scancel_cmd)

        if result["success"]:
            logger.info(f"Job {job_id} cancelled successfully")
        else:
            logger.error(f"Failed to cancel job {job_id}: {result['stderr']}")

        return result

    def list_jobs(self, user=None):
        """List SLURM jobs for a user.

        Parameters
        ----------
        user : str, optional
            Username to list jobs for.
                            If None, uses the SSH username.

        Returns
        -------
        dict: Dictionary containing:
            - jobs (list): List of job dictionaries with job information
            - success (bool): True if listing successful
        """
        if user is None:
            user = self.username

        squeue_cmd = (
            f"squeue --user={user} "
            + '--format="%i %T %j %u %P %M %l %D %R" --noheader'
        )
        result = self.execute_ssh_command(squeue_cmd)

        jobs = []
        if result["success"] and result["stdout"].strip():
            for line in result["stdout"].strip().split("\n"):
                fields = line.split()
                if len(fields) >= 6:
                    job_info = {
                        "job_id": fields[0],
                        "status": fields[1],
                        "job_name": fields[2],
                        "user": fields[3],
                        "partition": fields[4],
                        "time_used": fields[5],
                        "time_limit": fields[6] if len(fields) > 6 else "",
                        "nodes": fields[7] if len(fields) > 7 else "",
                        "reason": (
                            " ".join(fields[8:]) if len(fields) > 8 else ""
                        ),
                    }
                    jobs.append(job_info)

        return {
            "jobs": jobs,
            "success": result["success"],
            "stderr": result["stderr"],
        }

    def get_queue_info(self):
        """Get information about SLURM partitions/queues.

        Returns
        -------
        dict: Dictionary containing:
            - partitions (list): List of partition information dictionaries
            - success (bool): True if query successful
        """
        sinfo_cmd = 'sinfo --format="%P %a %l %D %T %N" --noheader'
        result = self.execute_ssh_command(sinfo_cmd)

        partitions = []
        if result["success"] and result["stdout"].strip():
            for line in result["stdout"].strip().split("\n"):
                fields = line.split()
                if len(fields) >= 5:
                    partition_info = {
                        "name": fields[0].rstrip(
                            "*"
                        ),  # Remove default partition marker
                        "availability": fields[1],
                        "time_limit": fields[2],
                        "nodes": fields[3],
                        "state": fields[4],
                        "node_list": (
                            " ".join(fields[5:]) if len(fields) > 5 else ""
                        ),
                    }
                    partitions.append(partition_info)

        return {
            "partitions": partitions,
            "success": result["success"],
            "stderr": result["stderr"],
        }
