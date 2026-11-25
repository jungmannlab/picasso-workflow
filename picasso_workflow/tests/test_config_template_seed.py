"""Unit tests for seeding the per-user config.yaml against a site config.

Covers the headless helpers that comment out template blocks already provided
by the site-wide (all-users) config, so the per-user file -- merged on top --
does not clobber the admin's settings.
"""

import picasso_workflow as pw

TEMPLATE = """\
Templates:
  dir: /somewhere

Confluence: # operational target
  url: https://example

SlurmLoginNodes:
  clusterA: login01

  clusterB: login02
Drivepaths:
  machine: /path
"""


def test_comment_out_commented_keys_only():
    out = pw._comment_out_template_keys(
        TEMPLATE, {"Confluence", "SlurmLoginNodes"}
    )
    lines = out.split("\n")
    # The targeted blocks are fully commented, including indented body lines.
    assert "# Confluence: # operational target" in lines
    assert "#   url: https://example" in lines
    assert "# SlurmLoginNodes:" in lines
    assert "#   clusterA: login01" in lines
    assert "#   clusterB: login02" in lines
    # A banner precedes each disabled block.
    assert any("uncomment to override" in ln for ln in lines)
    # Untouched blocks stay verbatim.
    assert "Templates:" in lines
    assert "  dir: /somewhere" in lines
    assert "Drivepaths:" in lines
    assert "  machine: /path" in lines


def test_blank_lines_within_block_preserved():
    out = pw._comment_out_template_keys(TEMPLATE, {"SlurmLoginNodes"})
    lines = out.split("\n")
    # The blank line between clusterA and clusterB stays blank (not "# ").
    i = lines.index("#   clusterA: login01")
    assert lines[i + 1] == ""  # preserved blank, part of the block
    assert lines[i + 2] == "#   clusterB: login02"


def test_block_boundary_stops_at_next_top_level_key():
    out = pw._comment_out_template_keys(TEMPLATE, {"SlurmLoginNodes"})
    # Drivepaths is the next top-level key -- it must remain active.
    assert "Drivepaths:" in out.split("\n")
    assert "# Drivepaths:" not in out.split("\n")


def test_empty_keys_returns_text_unchanged():
    assert pw._comment_out_template_keys(TEMPLATE, set()) == TEMPLATE
    assert pw._comment_out_template_keys(TEMPLATE, None) == TEMPLATE


def test_still_valid_yaml_after_commenting():
    import yaml

    out = pw._comment_out_template_keys(
        TEMPLATE, {"Confluence", "SlurmLoginNodes"}
    )
    data = yaml.safe_load(out)
    # Disabled keys are gone from the parsed result; others remain.
    assert "Confluence" not in data
    assert "SlurmLoginNodes" not in data
    assert data["Templates"] == {"dir": "/somewhere"}
    assert data["Drivepaths"] == {"machine": "/path"}


def test_site_config_keys_absent(monkeypatch, tmp_path):
    missing = tmp_path / "nope" / "config.yaml"
    monkeypatch.setattr(pw, "_site_config_path", lambda: missing)
    assert pw._site_config_keys() == set()


def test_site_config_keys_reads_top_level(monkeypatch, tmp_path):
    import yaml

    site = tmp_path / "config.yaml"
    site.write_text(
        yaml.safe_dump({"SlurmLoginNodes": {"a": "b"}, "Confluence": {}})
    )
    monkeypatch.setattr(pw, "_site_config_path", lambda: site)
    assert pw._site_config_keys() == {"SlurmLoginNodes", "Confluence"}
