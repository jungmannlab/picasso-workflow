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


def set_test_confluence_vars():
    """When loading the .env file in the GitHub Actions Runner from
    C:\\actions-runner\\.env via the .github\\workflows\\run-unittests.yml,
    the variables are in quotes. Remove these and set the variables anew.
    """
    confluence_url = os.getenv("TEST_CONFLUENCE_URL")
    confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
    confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
    confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")
    if confluence_url.startswith('"') and confluence_url.endswith('"'):
        confluence_url = confluence_url[1:-1]
        os.environ["TEST_CONFLUENCE_URL"] = confluence_url
    if confluence_token.startswith('"') and confluence_token.endswith('"'):
        confluence_token = confluence_token[1:-1]
        os.environ["TEST_CONFLUENCE_TOKEN"] = confluence_token
    if confluence_space.startswith('"') and confluence_space.endswith('"'):
        confluence_space = confluence_space[1:-1]
        os.environ["TEST_CONFLUENCE_SPACE"] = confluence_space
    if confluence_page.startswith('"') and confluence_page.endswith('"'):
        confluence_page = confluence_page[1:-1]
        os.environ["TEST_CONFLUENCE_PAGE"] = confluence_page


ensure_temp_folder()
set_test_confluence_vars()
