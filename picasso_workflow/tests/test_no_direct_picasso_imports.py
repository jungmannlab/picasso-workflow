#!/usr/bin/env python
"""Enforcement: production code imports picasso only through the boundary.

Every ``import picasso`` / ``from picasso import ...`` in picasso-workflow
production code must go through :mod:`picasso_workflow._picasso_api`, so the
picasso coupling stays a single reviewable surface and the contract test
(`test_picasso_contract.py`) stays authoritative. This test fails if a new
direct import sneaks in.

Detection is AST-based (not a text regex), so it is not fooled by `import
picasso, numpy`, `import  picasso` (odd whitespace), `import os; import
picasso`, and does not false-positive on picasso import examples that appear in
docstrings or comments. `picassosr` / `picasso_workflow` are not picasso.

The only production files allowed to import picasso directly are the boundary
module itself and the Qt-GUI tab (`render_tab.py`, which reaches `picasso.gui.*`
internals the flat boundary does not yet re-export). The private
``picasso.__main__`` symbols are reached only through guarded accessors in the
boundary, so they are NOT allowlisted anywhere else -- a new direct
`from picasso.__main__ import ...` outside the boundary now fails this test.
"""

import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parent.parent  # picasso_workflow/

# Production files allowed to import picasso directly, with the reason.
_ALLOWLIST_FILES = {
    "_picasso_api.py": "the declared boundary itself",
    "render_tab.py": "picasso.gui.* Qt internals not yet behind the boundary",
}


def _is_picasso(module: str) -> bool:
    """True for the picasso package itself, not picassosr / picasso_workflow."""
    return module == "picasso" or module.startswith("picasso.")


def _direct_picasso_imports(tree):
    """Yield (lineno, module) for every direct picasso import in an AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_picasso(alias.name):
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import (never picasso); module may be None
            if node.level == 0 and node.module and _is_picasso(node.module):
                yield node.lineno, node.module


def _production_files():
    for path in PKG.rglob("*.py"):
        if "tests" in path.relative_to(PKG).parts:
            continue
        yield path


def test_no_direct_picasso_imports_in_production():
    offenders = []
    for path in _production_files():
        if path.name in _ALLOWLIST_FILES:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for lineno, module in _direct_picasso_imports(tree):
            offenders.append(f"{path.relative_to(PKG)}:{lineno}: {module}")
    assert not offenders, (
        "Direct picasso imports found outside the boundary. Route these "
        "through `picasso_workflow._picasso_api` (see that module):\n  "
        + "\n  ".join(offenders)
    )
