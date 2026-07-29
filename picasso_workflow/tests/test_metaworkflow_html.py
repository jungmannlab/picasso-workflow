#!/usr/bin/env python
"""Tests for the coordinator's documentation-backend toggles.

A workflow coordinator can document to Confluence and/or to a local HTML
report. With Confluence off it must not connect (uses a no-op interface) and
the reporter config it builds must drop ``ConfluenceReporter`` and include
``HTMLReporter``.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from picasso_workflow.metaworkflow import (
    AggregationWorkflowCoordinator,
    SingleWorkflowCoordinator,
)
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


@patch("picasso_workflow.metaworkflow.time.sleep", lambda *a, **k: None)
@patch("picasso_workflow.metaworkflow.AggregationWorkflowRunner")
@patch("picasso_workflow.metaworkflow.io.load_info")
def test_prepare_analysis_shares_run_stamped_name_across_ranks(
    mock_load_info, mock_awr, tmp_path
):
    """Cooperating ranks must agree on one run-stamped parent-page name.

    Regression: for a single aggregation group whose src_loc carries no
    explicit ``report_name`` (only ``#tags``/``filepath``), the coordinator
    took an else branch that named the page after ``analysis_name`` alone and
    left each rank's runner to append its own ``datetime.now()`` postfix. Two
    ranks crossing a minute boundary then built ``..._1216`` vs ``..._1217``,
    and the worker rank looked for a parent page that never existed. The
    shared run timestamp must be baked into the name and threaded through as
    the runner postfix so every rank derives the identical name.
    """
    mock_load_info.return_value = [
        {
            "#tags": ["Cell 1_GFPNb", "Cell 1_NCR3-batch4"],
            "filepath": ["/data/a_locs.hdf5", "/data/b_locs.hdf5"],
        }
    ]
    fake_awr = MagicMock()
    fake_awr.result_folder = str(tmp_path)
    fake_awr.parameter_tiler = None
    mock_awr.config_from_dicts.return_value = fake_awr

    def make_coord(rank, size):
        coord = AggregationWorkflowCoordinator(
            "src.yaml",
            "260727-le",
            str(tmp_path),
            None,
            None,
            None,
            document_confluence=False,
        )
        # override the SLURM-derived (rank 0, size 1) identity
        coord.rank = rank
        coord.size = size
        return coord

    # One group, two ranks -> cooperative mode. Rank 0 (page owner) runs
    # first and publishes the shared stamp + page id; rank 1 is the worker.
    coord0 = make_coord(0, 2)
    coord0.prepare_analysis([("dummy_module", {})], [])
    coord1 = make_coord(1, 2)
    coord1.prepare_analysis([("dummy_module", {})], [])

    calls = mock_awr.config_from_dicts.call_args_list
    assert len(calls) == 2
    name0 = calls[0].args[0]["report_name"]
    name1 = calls[1].args[0]["report_name"]
    postfix0 = calls[0].kwargs["postfix"]
    postfix1 = calls[1].kwargs["postfix"]

    token = coord0._run_token()
    stamp = (
        (Path(coord0.root_folder) / f".pwf_runstamp_{token}")
        .read_text()
        .strip()
    )

    # identical name across ranks, carrying the shared run stamp and the
    # per-run token (which disambiguates concurrent runs of the same name) ...
    assert name0 == name1 == f"260727-le_{token}_{stamp}"
    # ... and the same stamp handed to the runner as its postfix, so it can
    # never fall back to a per-rank datetime.now().
    assert postfix0 == postfix1 == stamp
    # the owner published the parent page id under that shared name, so the
    # worker rank resolved it instead of racing a title lookup.
    assert Path(coord0._page_id_file(name0)).exists()
