#!/usr/bin/env python
"""Tests for the local HTML reporter.

Covers the Confluence storage-format -> HTML converter and an end-to-end
``HTMLReporter`` run that produces ``report.html`` and copies figures into
``assets/`` -- all without any Confluence connection.
"""

import os
from unittest.mock import patch, MagicMock

import yaml

from picasso_workflow.html_reporter import (
    HTMLReporter,
    storage_to_html,
    write_aggregation_index,
    regenerate_html_report,
)
from picasso_workflow.workflow import (
    WorkflowRunner,
    AggregationWorkflowRunner,
)

# CDATA delimiters are built by concatenation so no shell/parser mangles them.
_CDATA_OPEN = "<!" + "[CDATA["
_CDATA_CLOSE = "]]" + ">"


def test_storage_to_html_layout_image_expand_code():
    frag = (
        '<ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>'
        "<p><strong>Module 03: Rendering</strong></p>"
        "<ul><li>Duration: 1.2 s</li></ul>"
        '<ac:structured-macro ac:name="expand" ac:schema-version="1">'
        '<ac:parameter ac:name="title">Parameters</ac:parameter>'
        "<ac:rich-text-body><ul><li>box: 7</li>"
        "<li>a &amp; b</li></ul></ac:rich-text-body>"
        "</ac:structured-macro>"
        '<ac:image ac:height="450">'
        '<ri:attachment ri:filename="scene.png" /></ac:image>'
        '<ac:structured-macro ac:name="code" ac:schema-version="1">'
        '<ac:parameter ac:name="language">yaml</ac:parameter>'
        "<ac:plain-text-body>"
        + _CDATA_OPEN
        + 'a: 1\nb: "x<y"'
        + _CDATA_CLOSE
        + "</ac:plain-text-body></ac:structured-macro>"
        "</ac:layout-cell></ac:layout-section></ac:layout>"
    )
    out = storage_to_html(frag)

    # structural translations
    assert '<div class="cl-layout">' in out
    assert '<div class="cl-section cl-section-single">' in out
    assert '<details class="cl-expand"><summary>Parameters</summary>' in out
    assert '<img class="cl-img" src="assets/scene.png"' in out
    assert 'height="450"' in out
    assert '<pre class="cl-code"><code>' in out
    # plain HTML passes through; entities preserved
    assert "<li>box: 7</li>" in out
    assert "a &amp; b" in out
    # CDATA content is inlined as escaped text inside the code block
    assert "&lt;" in out  # the '<' from 'x<y'
    # no Confluence-only markup leaks through
    assert "<ac:" not in out
    assert "<ri:" not in out
    assert "CDATA" not in out


def test_storage_to_html_warning_admonition():
    """The Confluence warning macro becomes a styled callout box."""
    frag = (
        '<ac:structured-macro ac:name="warning">'
        "<ac:rich-text-body><p><strong>Warnings</strong></p>"
        "<ul><li>Channel 'gold' has 0 localization(s); skipping "
        "nearest-neighbour analysis.</li></ul>"
        "</ac:rich-text-body></ac:structured-macro>"
    )
    out = storage_to_html(frag)

    assert '<div class="cl-admonition cl-warning">' in out
    assert "has 0 localization(s)" in out
    assert "<ac:" not in out


def test_storage_to_html_multimedia():
    frag = (
        '<ac:structured-macro ac:name="multimedia" ac:schema-version="1">'
        '<ac:parameter ac:name="width">30%</ac:parameter>'
        '<ac:parameter ac:name="name">'
        '<ri:attachment ri:filename="movie.mp4" /></ac:parameter>'
        "</ac:structured-macro>"
    )
    out = storage_to_html(frag)
    assert '<video class="cl-video" src="assets/movie.mp4"' in out
    assert 'width="30"' in out
    assert "<ac:" not in out


def test_html_reporter_end_to_end(tmp_path):
    report_dir = str(tmp_path / "myreport")
    reporter = HTMLReporter(report_dir, "My Report")

    # a figure produced by the analysis side
    fig = tmp_path / "scene.png"
    fig.write_bytes(b"\x89PNG\r\n\x1a\n fake png bytes")

    # dummy_module needs only timing results
    reporter.dummy_module(
        0,
        {},
        {"start time": "t0", "end time": "t1", "duration": 65.0},
    )

    # exercise the image-attachment path
    reporter.ci.upload_attachment("local", str(fig))
    reporter.ci.update_page_content_with_image_attachment(
        "My Report", "local", "scene.png"
    )

    html_path = os.path.join(report_dir, "report.html")
    assert os.path.isfile(html_path)
    content = open(html_path, encoding="utf-8").read()

    # title, a module section, the TOC and the copied asset reference
    assert "My Report" in content
    assert "Dummy Module" in content
    assert 'src="assets/scene.png"' in content
    assert '<nav id="cl-toc">' in content
    # the figure was copied into assets/
    assert os.path.isfile(os.path.join(report_dir, "assets", "scene.png"))


def test_html_reporter_persists_sections_across_reopen(tmp_path):
    report_dir = str(tmp_path / "rep")
    r1 = HTMLReporter(report_dir, "Persist")
    r1.dummy_module(
        0, {}, {"start time": "t0", "end time": "t1", "duration": 1.0}
    )

    # re-open the same report dir (as a continued run would)
    r2 = HTMLReporter(report_dir, "Persist")
    r2.dummy_module(
        1, {}, {"start time": "t2", "end time": "t3", "duration": 2.0}
    )

    content = open(
        os.path.join(report_dir, "report.html"), encoding="utf-8"
    ).read()
    # both section blocks survive the reopen
    assert content.count('class="cl-block"') == 2


@patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
@patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
def test_workflowrunner_with_html_reporter_only(tmp_path):
    """A WorkflowRunner configured with HTMLReporter (and no Confluence)
    runs a module and produces report.html in the result folder."""
    reporter_config = {"report_name": "htmlrun", "HTMLReporter": {}}
    analysis_config = {"result_location": str(tmp_path)}

    wr = WorkflowRunner.config_from_dicts(reporter_config, analysis_config, [])

    # no Confluence reporter; exactly one (HTML) reporter is registered
    assert wr.confluencereporter is None
    assert len(wr.reporters) == 1

    # mocked analysis returns a result dict the dummy_module reporter can use
    wr.autopicasso.dummy_module = lambda i, p: (
        {},
        {
            "start time": "t0",
            "end time": "t1",
            "duration": 1.0,
            "success": True,
        },
    )
    success = wr.call_module("dummy_module", 0, {})
    assert success is True

    report_html = os.path.join(wr.result_folder, "report.html")
    assert os.path.isfile(report_html)
    assert "Dummy Module" in open(report_html, encoding="utf-8").read()


def test_write_aggregation_index(tmp_path):
    index_dir = str(tmp_path)
    child_reports = [
        ("dataset_00", "dataset_00/report.html", True),
        ("dataset_01", "dataset_01/report.html", False),  # missing
        ("Aggregation", "agg/report.html", True),
    ]
    rows = [("Report", "myagg"), ("Single datasets", 2)]
    path = write_aggregation_index(
        index_dir,
        "My Aggregation",
        rows,
        child_reports,
        config={"single_dataset_modules": [("load_dataset", {"a": 1})]},
    )
    assert os.path.isfile(path)
    content = open(path, encoding="utf-8").read()

    assert "My Aggregation" in content
    assert 'href="dataset_00/report.html"' in content
    assert 'href="agg/report.html"' in content
    # the missing report is listed but not linked
    assert 'href="dataset_01/report.html"' not in content
    assert "(no report)" in content
    # config snapshot present
    assert "Configuration (YAML)" in content
    assert "load_dataset" in content


@patch("picasso_workflow.workflow.ParameterTiler")
def test_aggregation_runner_writes_html_index(_mock_tiler, tmp_path):
    """An HTML-configured AggregationWorkflowRunner writes index.html that
    links each child report -- with no Confluence connection."""
    reporter_config = {"report_name": "aggrun", "HTMLReporter": {}}
    analysis_config = {"result_location": str(tmp_path)}
    aggregation_workflow = {
        "single_dataset_tileparameters": {},
        "single_dataset_modules": [],
        "aggregation_modules": [],
    }

    awr = AggregationWorkflowRunner.config_from_dicts(
        reporter_config, analysis_config, aggregation_workflow
    )
    assert awr._html_reporting is True

    # simulate two single-dataset reports plus an aggregation report
    sgl_folders = []
    for name in ["aggrun_sgl_00", "aggrun_sgl_01"]:
        folder = os.path.join(awr.result_folder, name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "report.html"), "w") as f:
            f.write("<html>child</html>")
        sgl_folders.append(folder)
    agg_folder = os.path.join(awr.result_folder, "aggrun_aggregation")
    os.makedirs(agg_folder, exist_ok=True)
    with open(os.path.join(agg_folder, "report.html"), "w") as f:
        f.write("<html>agg</html>")

    awr._write_html_overview(sgl_folders, agg_folder)

    index = os.path.join(awr.result_folder, "index.html")
    assert os.path.isfile(index)
    content = open(index, encoding="utf-8").read()
    assert 'href="aggrun_sgl_00/report.html"' in content
    assert 'href="aggrun_sgl_01/report.html"' in content
    assert 'href="aggrun_aggregation/report.html"' in content
    assert "Run overview" in content


def _write_single_runner_yaml(folder, report_name):
    """Write a minimal WorkflowRunner.yaml describing one dummy_module run."""
    os.makedirs(folder, exist_ok=True)
    data = {
        "results": {
            "00_dummy_module": {
                "start time": "t0",
                "end time": "t1",
                "duration": 1.0,
                "success": True,
            }
        },
        "reporter_config": {"report_name": report_name},
        "analysis_config": {},
        "workflow_modules": [("dummy_module", {})],
    }
    with open(os.path.join(folder, "WorkflowRunner.yaml"), "w") as f:
        yaml.dump(data, f)


def test_regenerate_single_from_folder(tmp_path):
    folder = str(tmp_path / "run_240101-1200")
    _write_single_runner_yaml(folder, "run")

    # no report.html yet (e.g. only Confluence was used originally)
    assert not os.path.isfile(os.path.join(folder, "report.html"))

    path = regenerate_html_report(folder)
    assert path == os.path.join(folder, "report.html")
    content = open(path, encoding="utf-8").read()
    assert "Dummy Module" in content


def test_upload_attachment_resolves_moved_figure(tmp_path):
    """A figure referenced by a stale absolute path is still found by
    basename within the (moved) result folder and copied into assets/."""
    report_dir = str(tmp_path / "run")
    reporter = HTMLReporter(report_dir, "run")

    # the real figure lives in a module subfolder
    sub = os.path.join(report_dir, "01_export_brightfield")
    os.makedirs(sub, exist_ok=True)
    with open(os.path.join(sub, "brightfield.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n fake")

    # the stored path points at a now-nonexistent original location
    stale = "/old/cluster/path/01_export_brightfield/brightfield.png"
    reporter.ci.upload_attachment("local", stale)

    assert os.path.isfile(
        os.path.join(report_dir, "assets", "brightfield.png")
    )


def test_regenerate_aggregation_from_folder(tmp_path):
    agg = str(tmp_path / "agg_240101-1400")
    os.makedirs(agg, exist_ok=True)
    # two child single-workflow folders
    for name in ["agg_sgl_00", "agg_aggregation"]:
        _write_single_runner_yaml(os.path.join(agg, name), name)
    # the aggregation runner state
    with open(os.path.join(agg, "AggregationWorkflowRunner.yaml"), "w") as f:
        yaml.dump(
            {
                "reporter_config": {"report_name": "agg"},
                "analysis_config": {},
                "aggregation_workflow": {"single_dataset_modules": []},
                "postfix": "240101-1400",
                "sgl_workflow_locations": [],
                "all_results": {},
            },
            f,
        )

    path = regenerate_html_report(agg)
    assert path == os.path.join(agg, "index.html")
    content = open(path, encoding="utf-8").read()
    assert 'href="agg_sgl_00/report.html"' in content
    assert 'href="agg_aggregation/report.html"' in content
    # children were actually regenerated
    assert os.path.isfile(os.path.join(agg, "agg_sgl_00", "report.html"))


def test_regenerate_raises_without_state(tmp_path):
    try:
        regenerate_html_report(str(tmp_path))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
