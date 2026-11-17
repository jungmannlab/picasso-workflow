import os


def temp_folder():
    """Return a writable folder for test result artefacts, creating it.

    Tests write their result folders under a shared ``temp/`` dir. In a source
    checkout that is the repo-root ``temp/`` (two levels up from this package),
    but when the package is installed **read-only** -- e.g. inside an Apptainer
    ``.sif`` -- that path is ``site-packages/temp`` and ``mkdir`` there raises
    ``OSError: Read-only file system``. Resolution order:

      1. ``$PW_TEST_TEMP_DIR`` (the cluster test chain sets this under the
         writable run dir).
      2. the repo-root ``temp/`` -- used when its parent is writable (source
         checkout, editable CI install).
      3. ``<cwd>/temp`` -- the fallback for a read-only install (the test chain
         runs pytest with the writable run dir as cwd).
    """
    env = os.getenv("PW_TEST_TEMP_DIR")
    if env:
        base = env
    else:
        repo_temp = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
            )
        )
        if os.access(os.path.dirname(repo_temp), os.W_OK):
            base = repo_temp
        else:
            base = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(base, exist_ok=True)
    return base


def ensure_temp_folder():
    """Ensure the shared test ``temp/`` folder exists (see :func:`temp_folder`).

    Kept for backwards compatibility; delegates to :func:`temp_folder`, which
    is read-only-filesystem safe.
    """
    temp_folder()


# Confluence test credentials are resolved on demand via
# picasso_workflow.confluence.resolve_confluence_credentials("ConfluenceTest")
# -- non-secret fields come from config.yaml's ConfluenceTest section (each
# overridable by a TEST_CONFLUENCE_* env var) and the token from the
# TEST_CONFLUENCE_TOKEN env var. The resolver strips any surrounding quotes,
# so no separate env-var sanitisation step is needed here.
ensure_temp_folder()
