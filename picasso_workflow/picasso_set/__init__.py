#!/usr/bin/env python
"""The picasso-set: modules recapitulating native picasso 1:1.

Each module here mirrors one picasso CLI command or library operation with
the exact same parameter names and defaults (documented deviations aside).
Module names carry the ``picasso_`` prefix so they are globally unique and
can be mixed freely with the classic module set in one workflow.

This package ``__init__`` is deliberately import-light: it must not import
``picasso``/``matplotlib``/``PyQt6``, because
:mod:`picasso_workflow.modulespec` (dependency-free by design) imports
:mod:`picasso_workflow.picasso_set.params`. The analysis mixins are imported
directly from their submodules (e.g.
``from picasso_workflow.picasso_set.analyse_core import PicassoSetCoreMixin``).

Author: Heinrich Grabmayr
Initial date: October 5, 2026
"""
