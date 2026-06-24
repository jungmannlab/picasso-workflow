#!/usr/bin/env python
"""Tests for the coordinator's documentation-backend toggles.

A workflow coordinator can document to Confluence and/or to a local HTML
report. With Confluence off it must not connect (uses a no-op interface) and
the reporter config it builds must drop ``ConfluenceReporter`` and include
``HTMLReporter``.
"""

from unittest.mock import patch

from picasso_workflow.metaworkflow import SingleWorkflowCoordinator
from picasso_workflow.confluence import NullConfluenceInterface


def test_coordinator_confluence_off_uses_html_and_no_network(tmp_path):
    # document_confluence=False must not touch the network at all
    coord = SingleWorkflowCoordinator(
        None,  # no input files -> no dataset loading
        "myrun",
        str(tmp_path),
        None,
        None,
        None,
        document_confluence=False,
    )
    # no real Confluence connection was made
    assert isinstance(coord.ci, NullConfluenceInterface)
    # HTML is auto-enabled when Confluence is off
    assert coord.document_html is True

    reporter_config, _ = coord.get_configs("myrun", str(tmp_path / "root"))
    assert "ConfluenceReporter" not in reporter_config
    assert "HTMLReporter" in reporter_config


@patch("picasso_workflow.confluence.ConfluenceInterface")
def test_coordinator_confluence_on_keeps_confluence(mock_ci, tmp_path):
    coord = SingleWorkflowCoordinator(
        None,
        "myrun",
        str(tmp_path),
        "http://confluence",
        "SPACE",
        "token",
        document_confluence=True,
    )
    # default: no HTML, Confluence used (mocked, so no real network)
    assert coord.document_html is False
    reporter_config, _ = coord.get_configs("myrun", str(tmp_path / "root"))
    assert "ConfluenceReporter" in reporter_config
    assert "HTMLReporter" not in reporter_config


@patch("picasso_workflow.confluence.ConfluenceInterface")
def test_coordinator_both_backends(mock_ci, tmp_path):
    coord = SingleWorkflowCoordinator(
        None,
        "myrun",
        str(tmp_path),
        "http://confluence",
        "SPACE",
        "token",
        document_confluence=True,
        document_html=True,
    )
    reporter_config, _ = coord.get_configs("myrun", str(tmp_path / "root"))
    assert "ConfluenceReporter" in reporter_config
    assert "HTMLReporter" in reporter_config


def test_null_confluence_interface_is_inert(tmp_path):
    ci = NullConfluenceInterface()
    # the method surface the coordinator uses returns benign values
    assert ci.create_page("page", "body") == "local"
    assert ci.get_page_properties("page") == ("local", "page")
    assert ci.get_page_body("page") == ""
    assert ci.update_page_content("p", "id", "body") is None
    assert ci.upload_attachment("id", "/some/fig.png") == "fig.png"


@patch("picasso_workflow.confluence.ConfluenceInterface")
def test_get_configs_threads_parent_page_id(mock_ci, tmp_path):
    """A given parent_page_id lands in the ConfluenceReporter config."""
    coord = SingleWorkflowCoordinator(
        None,
        "myrun",
        str(tmp_path),
        "http://confluence",
        "SPACE",
        "token",
        document_confluence=True,
    )
    reporter_config, _ = coord.get_configs(
        "myrun", str(tmp_path / "root"), parent_page_id="4242"
    )
    assert reporter_config["ConfluenceReporter"]["parent_page_id"] == "4242"


def test_publish_and_await_page_id_roundtrip(tmp_path):
    """A published page id is read back by _await_page_id (rank barrier)."""
    coord = SingleWorkflowCoordinator(
        None,
        "myrun",
        str(tmp_path),
        None,
        None,
        None,
        document_confluence=False,
    )
    coord.root_folder = str(tmp_path)

    coord._publish_page_id("260119_some-report_260623-1633", "778899")
    assert coord._await_page_id("260119_some-report_260623-1633") == "778899"


def test_publish_page_id_noop_for_missing_id(tmp_path):
    """Publishing a falsy id writes nothing (no file to read back)."""
    coord = SingleWorkflowCoordinator(
        None,
        "myrun",
        str(tmp_path),
        None,
        None,
        None,
        document_confluence=False,
    )
    coord.root_folder = str(tmp_path)

    coord._publish_page_id("report", None)
    import os

    assert not os.path.exists(coord._page_id_file("report"))
