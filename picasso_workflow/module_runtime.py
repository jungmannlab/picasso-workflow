#!/usr/bin/env python
"""Runtime plumbing shared by all module collections.

Holds :func:`module_decorator`, which wraps every analysis module of
:class:`~picasso_workflow.analyse.AutoPicasso` and of the picasso-set
mixins (:mod:`picasso_workflow.picasso_set`). It lives in its own module
so the mixin files can import it at class-definition time without
importing :mod:`picasso_workflow.analyse` (which imports the mixins).

Author: Heinrich Grabmayr
Initial date: October 5, 2026
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt


def module_decorator(method):
    """Wrap a module to manage its result folder and timing.

    Creates the module's result directory, seeds the ``results`` dict with
    ``folder`` and ``start time``, runs the module, then fills in ``success``
    (saving locs if requested), ``end time`` and ``duration``, and closes open
    figures.

    Parameters
    ----------
    method : callable
        The module method to wrap.

    Returns
    -------
    callable
        The wrapped method.
    """

    def module_wrapper(
        self, i, parameters, calling_module_dir=None, suffix=""
    ):
        # create the results direcotry
        # method_name = get_caller_name(2)
        method_name = method.__name__

        if calling_module_dir is None:
            module_result_dir = os.path.join(
                self.results_folder, f"{i:02d}_" + method_name + suffix
            )
        else:
            module_result_dir = os.path.join(
                calling_module_dir, f"{i:02d}_" + method_name + suffix
            )
        os.makedirs(module_result_dir, exist_ok=True)

        results = {
            "folder": os.path.normpath(module_result_dir),
            "start time": datetime.now().strftime("%y-%m-%d %H:%M:%S"),
        }

        # call the module. On failure, hand the partial results (folder and
        # start time) to the error reporter: they are built here and would
        # otherwise die with the stack frame, and the folder cannot be
        # reconstructed by the caller, which does not see suffix /
        # calling_module_dir.
        try:
            parameters, results = method(self, i, parameters, results)
        except BaseException as exc:
            exc._pwf_partial_results = results
            # Persist whatever locs the module held when it failed, so the
            # last data state can be inspected while debugging without
            # re-running the workflow. Best-effort: never mask the real error.
            self._save_state_on_error(results["folder"])
            raise

        # post-actions
        # modules only need to specifically set an error.
        if results.get("success") is None:
            results["success"] = True
            # save locs if desired
            if parameters.get("save_locs") is True or self.analysis_config.get(
                "always_save"
            ):
                # record what was written as a restore point ("checkpoint")
                # for resumed runs. A module that already saved its locs and
                # recorded them as a checkpoint (e.g. the picasso-set
                # modules, save_single_dataset) is not saved again: the
                # second write would be byte-identical.
                if (
                    hasattr(self, "locs")
                    and self.locs is not None
                    and results.get("checkpoint", {}).get("single") is None
                ):
                    fp = os.path.join(results["folder"], "locs.hdf5")
                    self._save_locs(fp)
                    results.setdefault("checkpoint", {})["single"] = {
                        "filepath": fp
                    }
                if (
                    hasattr(self, "channel_locs")
                    and self.channel_locs is not None
                ):
                    allfps = self._save_datasets_agg(results["folder"])
                    results.setdefault("checkpoint", {})["channels"] = {
                        "filepaths": allfps,
                        "tags": list(self.channel_tags),
                    }
        results["end time"] = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        td = datetime.strptime(
            results["end time"], "%y-%m-%d %H:%M:%S"
        ) - datetime.strptime(results["start time"], "%y-%m-%d %H:%M:%S")
        results["duration"] = td.total_seconds()
        # logger.debug(f"RESULTS: {results}")

        # close all figures potentially still open
        plt.close("all")
        return parameters, results

    return module_wrapper
