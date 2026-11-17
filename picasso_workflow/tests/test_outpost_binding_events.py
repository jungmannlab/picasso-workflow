#!/usr/bin/env python
"""The binding-events feature is importable at its new location.

Regression guard for the Phase 2 move of ``binding_event_analysis`` /
``binding_event_plotting`` into ``outpost/binding_events/`` and the fix to
``analyse.py``'s previously-broken
``outpost_modules.binding_event_analysis.Measurement`` reference (that submodule
was never imported into the ``outpost_modules`` package, so the call raised
``AttributeError`` at runtime -- but nothing exercised it, so no test caught it).

Needs the optional ``[binding-events]`` extra (``lmfit``, ``pandarallel``);
skipped when it is not installed.
"""

import pytest

pytest.importorskip("lmfit", reason="[binding-events] extra not installed")
pytest.importorskip(
    "pandarallel", reason="[binding-events] extra not installed"
)
# analysis.py imports picasso.postprocess through the _picasso_api boundary at
# module import, so the whole file needs a real picasso to collect.
pytest.importorskip("picasso", reason="picassosr not installed")


def test_binding_events_classes_importable():
    """The classes ``analyse.binding_event_analysis`` imports resolve."""
    from picasso_workflow.outpost.binding_events.analysis import Measurement
    from picasso_workflow.outpost.binding_events.plotting import Plotting

    assert callable(Measurement) and callable(Plotting)


def test_measurement_constructor_accepts_call_site_kwargs():
    """``Measurement(...)`` accepts exactly the kwargs ``analyse`` passes.

    Guards the ``saving_path=`` -> ``saving_name=`` fix: the call site used
    ``saving_path``, which the constructor never accepted (``TypeError``).
    """
    import inspect

    from picasso_workflow.outpost.binding_events.analysis import Measurement

    params = inspect.signature(Measurement.__init__).parameters
    for kwarg in (
        "in_path",
        "save_path",
        "saving_name",
        "total_n_frames",
        "pixelsize",
    ):
        assert kwarg in params, kwarg
    assert "saving_path" not in params


def _synthetic_grouped_table(n_groups=6, seed=1):
    """Build a synthetic clustered locs table.

    Columns ``group/frame/photons/sx/sy/bg`` drive SBR and the kinetics fits;
    ``x/y/lpx/lpy`` (each group a docking site clustered at its own centre, nm)
    let ``docking_site_damage`` link binding events instead of short-circuiting
    on a missing column. Each group gets enough frames across the acquisition
    for the rate-drop / docking-damage fits to have something to chew on.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        cx = float(rng.uniform(0, 40000))  # docking-site centre, nm
        cy = float(rng.uniform(0, 40000))
        frames = np.sort(
            rng.choice(np.arange(0, 20000), size=80, replace=False)
        )
        for fr in frames:
            rows.append(
                {
                    "group": g,
                    "frame": int(fr),
                    "photons": float(rng.uniform(1000, 5000)),
                    "sx": float(rng.uniform(80, 150)),  # nm (picasso 1.0)
                    "sy": float(rng.uniform(80, 150)),
                    "bg": float(rng.uniform(5, 50)),
                    "x": cx + float(rng.normal(0, 2.0)),  # nm
                    "y": cy + float(rng.normal(0, 2.0)),
                    "lpx": float(rng.uniform(1, 5)),  # nm
                    "lpy": float(rng.uniform(1, 5)),
                }
            )
    return pd.DataFrame(rows)


def test_begin_populates_sbr_zero_on_edge_frames(monkeypatch):
    """``Begin`` fills ``table_g['sbr']`` via ``signal_to_background``.

    SBR must be positive on center frames and exactly 0 on edge frames (the
    "0 on edge frames" semantics the retired hand-rolled integral produced).
    """
    from picasso_workflow.outpost.binding_events import analysis as amod

    table = _synthetic_grouped_table()
    m = amod.Measurement(total_n_frames=20000, pixelsize=130.0)
    # Feed the synthetic table instead of reading an hdf5 file.
    monkeypatch.setattr(m, "Import", lambda: setattr(m, "table", table))

    m.Begin()

    assert "sbr" in m.table_g.columns
    center = m.table_g["center_frame"].to_numpy(dtype=bool)
    sbr = m.table_g["sbr"].to_numpy(dtype=float)
    # Edge frames are masked to exactly 0; center frames are positive.
    assert (sbr[~center] == 0.0).all()
    assert (sbr[center] > 0.0).all()


def test_begin_sets_aggregate_metrics(monkeypatch):
    """``Begin`` always leaves ``loc_rate_drop`` / ``destroyed_fraction`` set.

    ``loc_rate_drop`` is a deterministic loc-count ratio (no fit), so on the
    populated synthetic table it must be finite -- a real positive-path check
    of the ``localization_rate_drop`` wiring. ``destroyed_fraction`` depends on
    an aggregate curve fit that may not converge on small synthetic data, so it
    is only required to be present (finite or NaN).
    """
    import math

    from picasso_workflow.outpost.binding_events import analysis as amod

    table = _synthetic_grouped_table()
    m = amod.Measurement(total_n_frames=20000, pixelsize=130.0)
    monkeypatch.setattr(m, "Import", lambda: setattr(m, "table", table))

    m.Begin()

    assert isinstance(m.loc_rate_drop, float)
    assert not math.isnan(m.loc_rate_drop)
    assert isinstance(m.destroyed_fraction, float)


def test_docking_site_damage_returns_fraction_destroyed():
    """picasso's dict carries the key ``Begin`` extracts.

    Pins the return-shape contract ``Begin`` depends on: it reads
    ``docking_site_damage(...)["fraction_destroyed"]`` and builds
    ``picked_locs`` by grouping on ``group``. A generous clustered table (many
    picks with ``x/y/lpx/lpy`` and many binding events each) reaches the
    aggregate fit rather than short-circuiting on an absent column, so a
    renamed key or a wrong ``picked_locs`` construction would fail here.
    """
    from picasso_workflow._picasso_api import postprocess

    table = _synthetic_grouped_table(n_groups=40)
    info = [{"Frames": 20000, "Pixelsize": 130.0}]
    picked_locs = [grp for _, grp in table.groupby("group")]

    result = postprocess.docking_site_damage(picked_locs, info)

    assert "fraction_destroyed" in result


def test_sbr_matches_signal_to_background_on_center_frames(monkeypatch):
    """The vectorized SBR equals ``signal_to_background`` on center frames.

    Guards the equivalence the migration relies on: on center frames
    ``center_photons == photons``, so the new per-row SBR must match picasso's
    ``signal_to_background`` exactly.
    """
    import numpy as np

    from picasso_workflow._picasso_api import postprocess
    from picasso_workflow.outpost.binding_events import analysis as amod

    table = _synthetic_grouped_table()
    m = amod.Measurement(total_n_frames=20000, pixelsize=130.0)
    monkeypatch.setattr(m, "Import", lambda: setattr(m, "table", table))

    m.Begin()

    expected = postprocess.signal_to_background(m.table_g, m.info)
    center = m.table_g["center_frame"].to_numpy(dtype=bool)
    got = m.table_g["sbr"].to_numpy(dtype=float)
    np.testing.assert_allclose(got[center], expected[center])
    assert (got[~center] == 0.0).all()


def test_cleaning_bg_removes_nonpositive_background():
    """cleaning_bg drops bg <= 0 rows (fit errors). Begin must assign its
    return (it does not mutate in place), else negatives reach the SBR.
    """
    import pandas as pd

    from picasso_workflow.outpost.binding_events.analysis import Measurement

    m = Measurement()
    m.table = pd.DataFrame(
        {"group": [0, 0, 1], "frame": [0, 1, 2], "bg": [5.0, -1.0, 0.0]}
    )
    cleaned = m.cleaning_bg()
    assert (cleaned["bg"] > 0).all()
    assert len(cleaned) == 1  # only the bg=5.0 row survives


def test_plotting_api_matches_call_site():
    """``Plotting`` exposes exactly the members ``analyse`` constructs/calls.

    Guards the wrong-case method fixes (``plot_td`` / ``plot_r`` /
    ``saveAllResults`` -> ``Plot_td`` / ``Plot_r`` / ``SaveAllResults``).
    """
    import inspect

    from picasso_workflow.outpost.binding_events.plotting import Plotting

    params = inspect.signature(Plotting.__init__).parameters
    for kwarg in (
        "table_g",
        "table_k",
        "show",
        "save",
        "saving_name",
        "total_n_frames",
    ):
        assert kwarg in params, kwarg

    for method in (
        "Plot_photons",
        "Plot_bg",
        "Plot_sbr",
        "Plot_tb",
        "Plot_td",
        "Plot_r",
        "Plot_locs",
        "SaveAllResults",
    ):
        assert callable(getattr(Plotting, method, None)), method


def test_measurement_methods_exist():
    """``Measurement`` exposes the methods ``analyse`` calls (``Begin`` /
    ``FileSaver``)."""
    from picasso_workflow.outpost.binding_events.analysis import Measurement

    for method in ("Begin", "FileSaver"):
        assert callable(getattr(Measurement, method, None)), method
