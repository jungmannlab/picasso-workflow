#!/usr/bin/env python
"""Local, navigable HTML report as an alternative to Confluence.

Defines :class:`HTMLReporter`, a drop-in sibling of
:class:`~picasso_workflow.confluence.ConfluenceReporter` that writes the same
per-module report content to a self-contained ``report.html`` (plus an
``assets/`` folder of figures) inside the run's result folder, with no
Confluence connection or credentials required.

It reuses the existing reporter methods unchanged: those methods build
Confluence storage-format strings and talk to ``self.ci``. Here ``self.ci`` is
an :class:`HTMLInterface` that mirrors the small slice of the
:class:`~picasso_workflow.confluence.ConfluenceInterface` API the reporter
uses, but converts the storage format to plain HTML and accumulates it into a
local page instead of posting to Confluence.

Author: Heinrich Grabmayr
Initial date: 2026
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
from html.parser import HTMLParser

import yaml
from loguru import logger

from picasso_workflow.confluence import ConfluenceReporter, _yaml_safe

# ---------------------------------------------------------------------------
# Confluence storage format -> HTML conversion
# ---------------------------------------------------------------------------

_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)

# Confluence admonition macros rendered as styled callout boxes in HTML.
_ADMONITIONS = ("warning", "note", "info", "tip")


class _StorageToHTML(HTMLParser):
    """Convert a Confluence storage-format fragment to plain HTML.

    Handles the finite set of Confluence constructs the reporter emits:
    ``ac:layout`` / ``ac:layout-section`` / ``ac:layout-cell`` (→ ``div``),
    ``ac:image`` + ``ri:attachment`` (→ ``img``), and ``ac:structured-macro``
    of type ``expand`` (→ ``details``), ``code`` (→ ``pre``) and
    ``multimedia`` (→ ``video``). All other (standard HTML) tags pass through.

    Parameters
    ----------
    asset_rel : str, optional
        Relative directory under which attachment files are referenced.
        Default is ``"assets"``.
    """

    def __init__(self, asset_rel="assets"):
        super().__init__(convert_charrefs=False)
        self.asset_rel = asset_rel
        self.out: list[str] = []
        # stack of structured-macro frames: {name, title, width, filename}
        self._macros: list[dict] = []
        self._param: str | None = None  # current ac:parameter name
        self._param_buf: list[str] = []
        self._image: dict | None = None  # set while inside ac:image

    # -- helpers ----------------------------------------------------------
    def _write(self, text: str) -> None:
        if self._param is None:
            self.out.append(text)
        # inside an ac:parameter the content is captured, not emitted

    def _emit_text(self, text: str) -> None:
        if self._param is not None:
            self._param_buf.append(text)
        else:
            self.out.append(text)

    @staticmethod
    def _passthrough_attrs(attrs) -> str:
        parts = []
        for k, v in attrs:
            if v is None:
                parts.append(f" {k}")
            else:
                parts.append(f' {k}="{html.escape(v, quote=True)}"')
        return "".join(parts)

    def _attachment_src(self, filename: str) -> str:
        return f"{self.asset_rel}/{filename}"

    # -- HTMLParser callbacks --------------------------------------------
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "ac:structured-macro":
            self._macros.append(
                {
                    "name": a.get("ac:name"),
                    "title": "",
                    "width": None,
                    "filename": None,
                }
            )
            return
        if tag == "ac:parameter":
            self._param = a.get("ac:name")
            self._param_buf = []
            return
        if tag == "ac:rich-text-body":
            if self._macros and self._macros[-1]["name"] == "expand":
                title = self._macros[-1].get("title") or "Details"
                self._write(
                    f'<details class="cl-expand"><summary>{title}</summary>'
                    '<div class="cl-expand-body">'
                )
            elif self._macros and self._macros[-1]["name"] in _ADMONITIONS:
                name = self._macros[-1]["name"]
                self._write(f'<div class="cl-admonition cl-{name}">')
            return
        if tag == "ac:plain-text-body":
            if self._macros and self._macros[-1]["name"] == "code":
                self._write('<pre class="cl-code"><code>')
            return
        if tag == "ac:image":
            self._image = {"height": a.get("ac:height")}
            return
        if tag == "ac:layout":
            self._write('<div class="cl-layout">')
            return
        if tag == "ac:layout-section":
            sect = a.get("ac:type", "")
            self._write(f'<div class="cl-section cl-section-{sect}">')
            return
        if tag == "ac:layout-cell":
            self._write('<div class="cl-cell">')
            return
        if tag == "ri:attachment":
            self._handle_attachment(a.get("ri:filename", ""))
            return
        if tag.startswith("ac:") or tag.startswith("ri:"):
            return  # drop any other confluence-only wrapper tag
        # standard HTML tag: pass through
        self._write(f"<{tag}{self._passthrough_attrs(attrs)}>")

    def handle_startendtag(self, tag, attrs):
        a = dict(attrs)
        if tag == "ri:attachment":
            self._handle_attachment(a.get("ri:filename", ""))
            return
        if tag.startswith("ac:") or tag.startswith("ri:"):
            return
        self._write(f"<{tag}{self._passthrough_attrs(attrs)} />")

    def handle_endtag(self, tag):
        if tag == "ac:structured-macro":
            macro = self._macros.pop() if self._macros else {}
            if macro.get("name") == "multimedia" and macro.get("filename"):
                width = macro.get("width") or "480"
                src = self._attachment_src(macro["filename"])
                self._write(
                    f'<video class="cl-video" src="{src}" controls '
                    f'width="{html.escape(str(width), quote=True)}"></video>'
                )
            return
        if tag == "ac:parameter":
            value = "".join(self._param_buf).strip()
            if self._macros:
                if self._param == "title":
                    self._macros[-1]["title"] = value
                elif self._param == "width":
                    self._macros[-1]["width"] = value.replace("%", "")
            self._param = None
            self._param_buf = []
            return
        if tag == "ac:rich-text-body":
            if self._macros and self._macros[-1]["name"] == "expand":
                self._write("</div></details>")
            elif self._macros and self._macros[-1]["name"] in _ADMONITIONS:
                self._write("</div>")
            return
        if tag == "ac:plain-text-body":
            if self._macros and self._macros[-1]["name"] == "code":
                self._write("</code></pre>")
            return
        if tag == "ac:image":
            self._image = None
            return
        if tag in ("ac:layout", "ac:layout-section", "ac:layout-cell"):
            self._write("</div>")
            return
        if tag.startswith("ac:") or tag.startswith("ri:"):
            return
        self._write(f"</{tag}>")

    def handle_data(self, data):
        self._emit_text(data)

    def handle_entityref(self, name):
        self._emit_text(f"&{name};")

    def handle_charref(self, name):
        self._emit_text(f"&#{name};")

    # -- internal ---------------------------------------------------------
    def _handle_attachment(self, filename: str) -> None:
        if not filename:
            return
        if self._image is not None:
            height = self._image.get("height")
            h = (
                f' height="{html.escape(str(height), quote=True)}"'
                if height
                else ""
            )
            self._write(
                f'<img class="cl-img" src="{self._attachment_src(filename)}"'
                f'{h} alt="{html.escape(filename, quote=True)}" '
                'loading="lazy" />'
            )
        elif self._macros and self._macros[-1]["name"] == "multimedia":
            self._macros[-1]["filename"] = filename
        else:
            src = self._attachment_src(filename)
            self._write(f'<a href="{src}">{html.escape(filename)}</a>')


def storage_to_html(storage_text: str, asset_rel: str = "assets") -> str:
    """Convert a Confluence storage-format fragment to plain HTML.

    Parameters
    ----------
    storage_text : str
        A (balanced) Confluence storage-format fragment, as built by the
        reporter module methods.
    asset_rel : str, optional
        Relative directory under which attachment files are referenced.
        Default is ``"assets"``.

    Returns
    -------
    str
        The equivalent plain HTML.
    """
    # CDATA cannot be parsed by html.parser reliably; extract it first and
    # inline it as escaped text (it only ever appears in code blocks).
    text = _CDATA_RE.sub(lambda m: html.escape(m.group(1)), storage_text)
    parser = _StorageToHTML(asset_rel=asset_rel)
    parser.feed(text)
    parser.close()
    return "".join(parser.out)


# ---------------------------------------------------------------------------
# Local HTML "interface" mirroring the used ConfluenceInterface API
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
:root {{ --fg:#1b1b1b; --muted:#666; --accent:#2a6; --border:#e0e0e0; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,
  sans-serif; color:var(--fg); line-height:1.45; }}
#cl-wrap {{ display:flex; align-items:flex-start; }}
#cl-toc {{ position:sticky; top:0; align-self:flex-start; width:260px;
  max-height:100vh; overflow:auto; padding:1.2rem 1rem; border-right:1px
  solid var(--border); font-size:0.9rem; background:#fafafa; }}
#cl-toc h2 {{ font-size:0.8rem; text-transform:uppercase; letter-spacing:
  0.05em; color:var(--muted); }}
#cl-toc ol {{ list-style:none; padding-left:0; margin:0; }}
#cl-toc li {{ margin:0.25rem 0; }}
#cl-toc a {{ text-decoration:none; color:var(--fg); }}
#cl-toc a:hover {{ color:var(--accent); }}
#cl-main {{ flex:1; padding:1.5rem 2rem; max-width:1100px; }}
#cl-main h1 {{ margin-top:0; }}
.cl-block {{ border-top:1px solid var(--border); padding-top:0.5rem;
  margin-top:1.5rem; scroll-margin-top:1rem; }}
.cl-layout, .cl-section {{ display:flex; gap:1rem; flex-wrap:wrap; }}
.cl-cell {{ flex:1; min-width:0; }}
.cl-img {{ max-width:100%; height:auto; }}
.cl-video {{ max-width:100%; }}
table {{ border-collapse:collapse; }}
th, td {{ text-align:left; vertical-align:top; padding:0.2rem 0.5rem; }}
.cl-code {{ background:#f6f8fa; padding:0.75rem; overflow:auto; font-size:
  0.85rem; border-radius:4px; }}
details.cl-expand {{ margin:0.5rem 0; }}
details.cl-expand > summary {{ cursor:pointer; color:var(--accent); }}
.cl-meta {{ color:var(--muted); font-size:0.85rem; }}
.cl-admonition {{ margin:0.75rem 0; padding:0.5rem 0.9rem; border-left:4px
  solid var(--border); border-radius:4px; background:#fafafa; }}
.cl-admonition.cl-warning {{ border-left-color:#e0a800; background:#fff8e6; }}
.cl-admonition.cl-note {{ border-left-color:#6c757d; background:#f1f3f5; }}
.cl-admonition.cl-info {{ border-left-color:#2a6; background:#eafaf1; }}
.cl-admonition.cl-tip {{ border-left-color:#2a6; background:#eafaf1; }}
</style>
</head>
<body>
<div id="cl-wrap">
<nav id="cl-toc"><h2>Contents</h2><ol>{toc}</ol></nav>
<main id="cl-main">
<h1>{title}</h1>
<p class="cl-meta">Generated by picasso-workflow.</p>
{body}
</main>
</div>
</body>
</html>
"""

_HEADING_RE = re.compile(r"<strong>(.*?)</strong>", re.IGNORECASE | re.DOTALL)


class HTMLInterface:
    """Local HTML sink mirroring the used ``ConfluenceInterface`` API.

    Accumulates converted report sections into a single navigable
    ``report.html`` and copies referenced figures into an ``assets/`` folder.
    Section state is persisted to a sidecar JSON so that re-opening the report
    (e.g. when continuing a previous run) preserves earlier sections.

    Parameters
    ----------
    report_dir : str
        Directory to write ``report.html`` and ``assets/`` into.
    report_name : str
        Title shown on the report page.
    fresh : bool, optional
        Start from an empty report, ignoring (and overwriting) any existing
        sidecar. Used when regenerating a report from scratch. Default is
        False, which preserves earlier sections.
    """

    def __init__(self, report_dir: str, report_name: str, fresh=False):
        self.report_dir = report_dir
        self.report_name = report_name
        self.assets_dir = os.path.join(report_dir, "assets")
        self.html_path = os.path.join(report_dir, "report.html")
        self._sidecar = os.path.join(report_dir, ".report_sections.json")
        os.makedirs(self.assets_dir, exist_ok=True)
        # (anchor_id, toc_label, html) per appended section
        self.sections: list[list[str]] = []
        # lazily-built {basename: path} index for resolving moved figures
        self._asset_index: dict | None = None
        if not fresh and os.path.exists(self._sidecar):
            try:
                with open(self._sidecar) as f:
                    self.sections = json.load(f)
            except Exception as e:  # corrupt sidecar must not block a run
                logger.debug(f"Could not read report sidecar: {e}")
        self._render()

    # -- ConfluenceInterface-compatible surface ---------------------------
    def create_page(self, page_title, body_text="", parent_id="rootparent"):
        """Initialize the report page (mirrors ``create_page``)."""
        self.report_name = page_title
        if body_text:
            self._append(body_text)
        else:
            self._render()
        return "local"

    def update_page_content(
        self, page_name, page_id, body_update, replace=False
    ):
        """Append (or replace) a report section (mirrors Confluence)."""
        if replace:
            self.sections = []
        self._append(body_update)
        return "local"

    def upload_attachment(self, page_id, filename):
        """Copy a figure/file into ``assets/`` (mirrors Confluence)."""
        src = self._resolve_source(filename)
        if src is None:
            logger.debug(f"Attachment not found, skipping: {filename}")
            return os.path.basename(filename)
        try:
            dest = os.path.join(self.assets_dir, os.path.basename(filename))
            if os.path.abspath(src) != os.path.abspath(dest):
                shutil.copy2(src, dest)
        except OSError as e:
            logger.debug(f"Could not copy attachment {filename}: {e}")
        return os.path.basename(filename)

    def _resolve_source(self, filename: str) -> str | None:
        """Resolve a (possibly stale) attachment path to a real file.

        Returns ``filename`` if it exists; otherwise -- e.g. when a result
        folder has been moved/copied so the stored absolute paths no longer
        resolve -- searches ``report_dir`` for a file of the same basename.
        """
        if filename and os.path.isfile(filename):
            return filename
        base = os.path.basename(filename)
        if self._asset_index is None:
            self._asset_index = {}
            for root, _, files in os.walk(self.report_dir):
                if os.path.abspath(root) == os.path.abspath(self.assets_dir):
                    continue  # don't resolve to a previous copy in assets/
                for fn in files:
                    self._asset_index.setdefault(fn, os.path.join(root, fn))
        return self._asset_index.get(base)

    def update_page_content_with_image_attachment(
        self, page_name, page_id, filename
    ):
        """Append an image section by basename (mirrors Confluence)."""
        fn = os.path.basename(filename)
        self._append(
            f'<ac:image ac:height="350"><ri:attachment ri:filename="{fn}" />'
            "</ac:image>"
        )

    def update_page_content_with_movie_attachment(
        self, page_name, page_id, filename
    ):
        """Append a video section by basename (mirrors Confluence)."""
        fn = os.path.basename(filename)
        self._append(
            '<ac:structured-macro ac:name="multimedia">'
            '<ac:parameter ac:name="width">480</ac:parameter>'
            '<ac:parameter ac:name="name">'
            f'<ri:attachment ri:filename="{fn}" /></ac:parameter>'
            "</ac:structured-macro>"
        )

    # -- benign stubs for the rest of the API -----------------------------
    def get_page_properties(self, page_title="", page_id=""):
        return "local", page_title or self.report_name

    def get_page_version(self, page_title="", page_id=""):
        return 1

    def get_page_body(self, page_title="", page_id=""):
        return "".join(s[2] for s in self.sections)

    def get_attachment_id(self, page_id, filename):
        return os.path.basename(filename)

    def delete_attachment(self, page_id, attachment_id):
        pass

    def delete_page(self, page_id, recursive=False):
        pass

    # -- internals --------------------------------------------------------
    def _append(self, storage_text: str) -> None:
        body_html = storage_to_html(storage_text, asset_rel="assets")
        anchor = f"sec-{len(self.sections)}"
        label = self._section_label(body_html, len(self.sections))
        self.sections.append([anchor, label, body_html])
        self._render()

    @staticmethod
    def _section_label(body_html: str, index: int) -> str:
        match = _HEADING_RE.search(body_html)
        if match:
            text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            if text:
                return text
        return f"Section {index + 1}"

    def _render(self) -> None:
        toc = "".join(
            f'<li><a href="#{anchor}">{html.escape(label)}</a></li>'
            for anchor, label, _ in self.sections
        )
        body = "".join(
            f'<section class="cl-block" id="{anchor}">{section_html}</section>'
            for anchor, _, section_html in self.sections
        )
        doc = _PAGE_TEMPLATE.format(
            title=html.escape(self.report_name), toc=toc, body=body
        )
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(doc)
        try:
            with open(self._sidecar, "w") as f:
                json.dump(self.sections, f)
        except OSError as e:
            logger.debug(f"Could not write report sidecar: {e}")


def write_aggregation_index(
    index_dir: str,
    title: str,
    rows: list,
    child_reports: list,
    config=None,
) -> str:
    """Write a top-level ``index.html`` for an aggregation run.

    Produces a navigable overview page that shows run metadata and an optional
    YAML configuration snapshot, and links to each child report (the
    per-dataset single-workflow reports and the aggregation report).

    Parameters
    ----------
    index_dir : str
        Directory to write ``index.html`` into (the aggregation result
        folder).
    title : str
        Page title.
    rows : list of tuple
        ``(label, value)`` run-metadata pairs rendered as a table.
    child_reports : list of tuple
        ``(label, href, exists)`` per child report. ``href`` is relative to
        ``index_dir``; ``exists`` controls whether it is rendered as a link.
    config : object, optional
        Run configuration rendered as a collapsible YAML snapshot.

    Returns
    -------
    str
        The path to the written ``index.html``.
    """
    os.makedirs(index_dir, exist_ok=True)

    table = "".join(
        f"<tr><th>{html.escape(str(k))}</th>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    body = f"<h2>Run overview</h2><table><tbody>{table}</tbody></table>"

    if config is not None:
        try:
            snap = yaml.safe_dump(
                _yaml_safe(config), sort_keys=False, allow_unicode=True
            )
            body += (
                '<details class="cl-expand"><summary>Configuration (YAML)'
                '</summary><pre class="cl-code"><code>'
                f"{html.escape(snap)}</code></pre></details>"
            )
        except Exception as e:  # a snapshot issue must not block the index
            logger.debug(f"Could not serialize aggregation config: {e}")

    items = []
    for label, href, exists in child_reports:
        if exists:
            items.append(
                f'<li><a href="{html.escape(href, quote=True)}">'
                f"{html.escape(label)}</a></li>"
            )
        else:
            items.append(
                f"<li>{html.escape(label)} "
                '<span class="cl-meta">(no report)</span></li>'
            )
    body += f"<h2>Reports</h2><ul>{''.join(items)}</ul>"

    toc = "".join(
        f'<li><a href="{html.escape(href, quote=True)}">'
        f"{html.escape(label)}</a></li>"
        for label, href, exists in child_reports
        if exists
    )
    doc = _PAGE_TEMPLATE.format(title=html.escape(title), toc=toc, body=body)
    path = os.path.join(index_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


class HTMLReporter(ConfluenceReporter):
    """Reporter that writes a local navigable HTML report.

    A drop-in sibling of :class:`ConfluenceReporter`: it inherits all the
    per-module reporter methods unchanged and only swaps the ``self.ci`` sink
    for an :class:`HTMLInterface`, so no Confluence connection is made.

    Parameters
    ----------
    report_dir : str
        Directory to write ``report.html`` and ``assets/`` into.
    report_name : str
        Title of the report.
    fresh : bool, optional
        Start from an empty report (used when regenerating). Default is False.
    """

    def __init__(
        self, report_dir: str, report_name: str, fresh=False, **kwargs
    ):
        logger.debug(f"Initializing HTMLReporter at {report_dir}.")
        self.ci = HTMLInterface(report_dir, report_name, fresh=fresh)
        self.report_page_name = report_name
        self.report_page_id = self.ci.create_page(report_name, body_text="")
        self.report_dir = report_dir


# ---------------------------------------------------------------------------
# Regenerate reports from a saved result folder (no analysis re-run)
# ---------------------------------------------------------------------------


class _RunnerLoader(yaml.SafeLoader):
    """Safe YAML loader that also reconstructs ``python/tuple`` nodes."""


_RunnerLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def _load_runner_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_RunnerLoader)


def load_runner_state(path: str) -> dict:
    """Load a saved runner state file (``*WorkflowRunner.yaml``).

    Public, tuple-tolerant reader for the persisted run state, e.g. for a UI
    that wants to show per-module status without re-running anything.

    Parameters
    ----------
    path : str
        Path to a ``WorkflowRunner.yaml`` or
        ``AggregationWorkflowRunner.yaml`` file.

    Returns
    -------
    dict
        The parsed state.
    """
    return _load_runner_yaml(path)


def _regenerate_single(result_folder: str) -> str:
    """Rebuild ``report.html`` for a single-dataset result folder."""
    data = _load_runner_yaml(
        os.path.join(result_folder, "WorkflowRunner.yaml")
    )
    results = data.get("results") or {}
    workflow_modules = data.get("workflow_modules") or []
    report_name = (data.get("reporter_config") or {}).get(
        "report_name"
    ) or os.path.basename(result_folder.rstrip(os.sep))

    reporter = HTMLReporter(result_folder, report_name, fresh=True)
    for i, item in enumerate(workflow_modules):
        name = item[0]
        params = item[1] if len(item) > 1 else {}
        res = results.get(f"{i:02d}_{name}")
        if res is None:
            continue  # module never ran
        method = getattr(reporter, name, None)
        if method is None:
            continue
        try:
            method(i, params, res)
        except Exception as e:  # a single bad section must not abort the rest
            logger.warning(
                f"regenerate: reporter for {i:02d}_{name} failed: {e}"
            )
    return reporter.ci.html_path


def _regenerate_aggregation(result_folder: str) -> str:
    """Rebuild child reports + ``index.html`` for an aggregation folder."""
    data = _load_runner_yaml(
        os.path.join(result_folder, "AggregationWorkflowRunner.yaml")
    )
    report_name = (data.get("reporter_config") or {}).get(
        "report_name"
    ) or os.path.basename(result_folder.rstrip(os.sep))

    # Regenerate every child single-workflow folder found beneath this one.
    child_dirs = sorted(
        os.path.join(result_folder, d)
        for d in os.listdir(result_folder)
        if os.path.isfile(
            os.path.join(result_folder, d, "WorkflowRunner.yaml")
        )
    )
    child_reports = []
    for d in child_dirs:
        try:
            _regenerate_single(d)
        except Exception as e:
            logger.warning(f"regenerate: child {d} failed: {e}")
        report = os.path.join(d, "report.html")
        href = os.path.relpath(report, result_folder)
        child_reports.append(
            (os.path.basename(d), href, os.path.isfile(report))
        )

    rows = [
        ("Report", report_name),
        ("Result folder", result_folder),
        ("Child reports", len(child_reports)),
        ("Reports found", sum(1 for _, _, ok in child_reports if ok)),
    ]
    return write_aggregation_index(
        result_folder,
        report_name,
        rows,
        child_reports,
        config=data.get("aggregation_workflow"),
    )


def regenerate_html_report(result_folder: str) -> str:
    """Regenerate the HTML report(s) for a saved result folder.

    Replays the reporter from the persisted run state (no analysis is
    re-run). Works for both a single-dataset folder (``WorkflowRunner.yaml``
    → ``report.html``) and an aggregation folder
    (``AggregationWorkflowRunner.yaml`` → child ``report.html`` files plus a
    top-level ``index.html``). Figures are resolved even if the folder has
    been moved, by basename within ``result_folder``.

    Parameters
    ----------
    result_folder : str
        A run's result folder.

    Returns
    -------
    str
        Path to the top-level report (``index.html`` for an aggregation run,
        otherwise ``report.html``).

    Raises
    ------
    FileNotFoundError
        If no runner state file is found in ``result_folder``.
    """
    if os.path.isfile(
        os.path.join(result_folder, "AggregationWorkflowRunner.yaml")
    ):
        return _regenerate_aggregation(result_folder)
    if os.path.isfile(os.path.join(result_folder, "WorkflowRunner.yaml")):
        return _regenerate_single(result_folder)
    raise FileNotFoundError(
        "No WorkflowRunner.yaml or AggregationWorkflowRunner.yaml in "
        f"{result_folder}"
    )
