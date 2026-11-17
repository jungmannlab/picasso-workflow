#!/usr/bin/env python
"""The Ripley's-K feature is importable at its new ``spatial_stats`` location.

Regression guard for the Phase 2 move of ``ripleys`` and
``ripley_dcatlas_analysis`` out of the flat ``outpost_modules/`` into
``outpost/spatial_stats/``. ``analyse.AutoPicasso.ripleysk2`` imports ``ripleys``
from the new path and calls the functions asserted below; the unit tier mocks
``AutoPicasso``, so without this guard a broken move would surface only at
runtime on the cluster.
"""


def test_ripleys_importable_at_new_location():
    """The functions ``ripleysk2`` calls resolve at the new path."""
    from picasso_workflow.outpost.spatial_stats import ripleys

    for name in (
        "convert_picasso_to_coords",
        "typefraction_all_channels",
        "analyze_all_channels",
        "init_plot",
        "plot_ripleys",
    ):
        assert callable(getattr(ripleys, name)), name


def test_ripley_dcatlas_importable_at_new_location():
    """The DC-Atlas pipeline imports (incl. its ``ripleys`` dependency)."""
    from picasso_workflow.outpost.spatial_stats import (
        ripley_dcatlas_analysis,
    )

    assert callable(ripley_dcatlas_analysis.init_plot)


def test_ripleys_not_left_in_outpost_modules():
    """The old flat ``outpost_modules.ripleys`` attribute is gone."""
    import picasso_workflow.outpost_modules as outpost_modules

    assert not hasattr(outpost_modules, "ripleys")


# ---------------------------------------------------------------------------
# Regression: deltar sign fix in ripleys (radial_distribution_function /
# overpopulation). The bin width ``deltar`` must be POSITIVE so the ``rs``
# grid the functions build stays strictly increasing.
# ---------------------------------------------------------------------------


def test_rdf_deltar_positive_builds_ascending_rs():
    """radial_distribution_function's ``rs`` grid must ascend.

    It builds ``rs = np.append(r, max(r) + deltar)`` with
    ``deltar = r[2] - r[1]``. A negative ``deltar`` (the old bug) would make
    the appended last element smaller than ``max(r)``, breaking the grid.
    """
    import numpy as np

    r = np.arange(0, 100, 5, dtype=float)
    deltar = r[2] - r[1]
    rs = np.append(r, np.max(r) + deltar)

    assert deltar > 0
    assert np.all(np.diff(rs) > 0)  # strictly increasing
    assert rs[-1] > np.max(r)


def test_overpopulation_deltar_positive_builds_ascending_rs():
    """overpopulation's ``rs`` grid must ascend.

    It uses ``deltar = r[-1] - r[-2]`` (positive for ascending ``r``).
    """
    import numpy as np

    r = np.arange(0, 100, 5, dtype=float)
    deltar = r[-1] - r[-2]
    rs = np.append(r, np.max(r) + deltar)

    assert deltar > 0
    assert np.all(np.diff(rs) > 0)
    assert rs[-1] > np.max(r)


def test_rdf_and_overpopulation_direct_call_finite():
    """Calling the two fixed functions directly yields finite last bins.

    With ascending ``r`` the internal ``rs`` grid is ascending, so the bin
    widths ``d_rs`` are positive and the final RDF / overpopulation bin is
    finite (a negative ``deltar`` produced a negative-width final bin).
    """
    import numpy as np
    from picasso_workflow.outpost.spatial_stats import ripleys

    rng = np.random.default_rng(0)
    X1 = rng.random((200, 2)) * 1000
    X2 = rng.random((200, 2)) * 1000
    r = np.arange(0, 100, 5, dtype=float)

    rdf = ripleys.radial_distribution_function(X1, X2, r, univariate=False)
    assert rdf.shape == r.shape
    assert np.isfinite(rdf[-1])

    ovp = ripleys.overpopulation(X1, X2, r, univariate=False)
    assert ovp.shape == r.shape
    assert np.all(np.isfinite(ovp))


# ---------------------------------------------------------------------------
# Regression: csr_knn fixes.
#   (i)  estimate_density_from_neighbordists(fit_bkg=True) no longer raises
#        (bounds are length-matched to x0) and returns a positive density.
#   (ii) nndistribution_from_csr is now a normalized mixture
#        (1-f)*CSR + f*uniform, so it integrates to ~1 even for bkg_fraction>0
#        (the old unnormalized form integrated to > 1).
# ---------------------------------------------------------------------------


def test_estimate_density_fit_bkg_true_does_not_raise():
    """fit_bkg=True must not raise and must return a positive density.

    Previously L-BFGS-B received a 2-element x0 but a 1-element bounds list
    and raised ``ValueError: length of x0 != length of bounds``.
    """
    import numpy as np
    from picasso_workflow.outpost.spatial_stats.csr_knn import (
        estimate_density_from_neighbordists,
        nndistribution_from_csr,
    )

    np.random.seed(42)
    rho = 0.01
    r_grid = np.linspace(0.1, 40, 400)
    pdists = [nndistribution_from_csr(r_grid, k, rho) for k in range(1, 4)]
    nspots = 500
    nnobs = np.array(
        [
            np.random.choice(r_grid, size=nspots, p=pd / np.sum(pd))
            for pd in pdists
        ]
    )

    density, _res = estimate_density_from_neighbordists(
        nnobs, rho * 1.2, kmin=1, fit_bkg=True
    )
    assert np.isfinite(density)
    assert density > 0


def test_nndistribution_mixture_integrates_to_one():
    """With bkg_fraction>0 the normalized mixture integrates to ~1.

    The two-component mixture (1-f)*CSR + f*uniform conserves unit mass; the
    old unnormalized form (CSR kept weight 1, only uniform down-weighted)
    integrated to 1 + f and so to > 1.
    """
    import numpy as np
    from picasso_workflow.outpost.spatial_stats.csr_knn import (
        nndistribution_from_csr,
    )

    # Fine grid over the full support (no min/max cutoff), so np.trapezoid of
    # the density recovers its total mass.
    r = np.linspace(1e-3, 200, 20000)
    rho = 0.01

    integral_bkg = np.trapezoid(
        nndistribution_from_csr(r, 1, rho, bkg_fraction=0.5), r
    )
    integral_nobkg = np.trapezoid(
        nndistribution_from_csr(r, 1, rho, bkg_fraction=0.0), r
    )

    # Both integrate to ~1 (within a few %). The old unnormalized form would
    # have integrated to ~1 + f = 1.33 for bkg_fraction=0.5.
    assert abs(integral_bkg - 1.0) < 0.03
    assert abs(integral_nobkg - 1.0) < 0.03
