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


# Each test resolves (and creates) its result dir via `temp_folder()` in setUp
# / at module scope, so the tests package has no import-time side effects -- an
# eager `os.mkdir` here was what crashed pytest collection on a read-only
# Apptainer install.
#
# Confluence test credentials are resolved on demand via
# picasso_workflow.confluence.resolve_confluence_credentials("ConfluenceTest")
# -- non-secret fields come from config.yaml's ConfluenceTest section (each
# overridable by a TEST_CONFLUENCE_* env var) and the token from the
# TEST_CONFLUENCE_TOKEN env var. The resolver strips any surrounding quotes,
# so no separate env-var sanitisation step is needed here.
