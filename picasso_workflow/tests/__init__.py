import os


def ensure_temp_folder():
    """For unittesting, especially on a GitHub Actions Runner, ensure
    the temp folder exists.
    """
    results_folder = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        )
    )
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


# Confluence test credentials are resolved on demand via
# picasso_workflow.confluence.resolve_confluence_credentials("ConfluenceTest")
# -- non-secret fields come from config.yaml's ConfluenceTest section (each
# overridable by a TEST_CONFLUENCE_* env var) and the token from the
# TEST_CONFLUENCE_TOKEN env var. The resolver strips any surrounding quotes,
# so no separate env-var sanitisation step is needed here.
ensure_temp_folder()
