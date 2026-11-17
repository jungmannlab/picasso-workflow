"""Rendering feature group (Phase 2 restructure).

Holds picasso-workflow's rendering glue moved out of the flat
``outpost_modules/``:

* ``render`` -- matplotlib wrappers over ``picasso.render`` (``plot_scene`` and
  the single-/multi-channel scene renderers, colour helpers, contrast scaling).

Workflow-side presentation glue (`keep`) over the upstream ``picasso.render``
backend, which it reaches through ``_picasso_api`` -- see ``../MANIFEST.md``.
"""
