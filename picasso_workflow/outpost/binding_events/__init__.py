"""Binding-event analysis feature group (Phase 2 restructure).

DNA-PAINT binding-event analysis + plotting (Steen, Unterauer, Masullo et al.,
*Nat Methods* 2024), moved out of the flat ``outpost_modules/``:

* ``analysis``  -- the ``Measurement`` class (kinetics/photon/SBR extraction).
* ``plotting``  -- the ``Plotting`` class (per-metric figures + results export).

Self-contained (no picasso/sibling coupling). Requires the optional
``[binding-events]`` extra (``lmfit``, ``pandarallel``); imported lazily by the
``binding_event_analysis`` workflow module, so the package import stays cheap.
See ``../MANIFEST.md`` (a `promote` candidate).
"""
