"""SPINNA wrappers over ``picasso.spinna`` (extracted from picasso_outpost).

See the package docstring. `keep` -- workflow-side orchestration + plots.
"""

import os
import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from picasso_workflow._picasso_api import (
    spinna,
    get_internal,
    get_spinna_batch_analysis,
)


def spinna_batch(parameters_filename):
    """Run a SPINNA batch analysis from a config file.

    Mirrors the command-line entry point in ``picasso.__main__``. picasso's
    ``_spinna_batch_analysis`` derives the result directory from the
    parameters filename (appending an index if such a directory already
    exists) and saves the summary csv and NND figures there but returns no
    paths, so they are reconstructed here with the same logic.

    Parameters
    ----------
    parameters_filename : str
        Filepath of the SPINNA batch-analysis config (``.csv``) file.

    Returns
    -------
    result_dir : str
        Folder containing the results.
    fp_summary : str
        Filepath of the summary csv file.
    fp_fig : list of str
        Filepaths of the NND figures.
    """
    result_dir = parameters_filename.replace(".csv", "_fitting_results")
    if os.path.isdir(result_dir):
        i = 1
        while os.path.isdir(f"{result_dir}_{i}"):
            i += 1
        result_dir = f"{result_dir}_{i}"

    # Private picasso symbol, reached only through the declared boundary in
    # picasso_workflow._picasso_api (see that module).
    spinna_batch_analysis = get_spinna_batch_analysis()
    spinna_batch_analysis(parameters_filename)

    fp_summary = os.path.join(result_dir, "summary_results.csv")
    fp_fig = sorted(glob.glob(os.path.join(result_dir, "*_NND_*.png")))
    return result_dir, fp_summary, fp_fig


def _plot_label_unc_scan(
    target,
    candidates,
    scores,
    best_idx,
    save_filename,
    result_dir,
):
    """Plot the labeling-uncertainty screening curve for one target.

    Parameters
    ----------
    target : str
        Molecular target name.
    candidates : list of float
        Screened labeling uncertainties in nm.
    scores : list of float
        Kolmogorov-Smirnov fit score for each candidate.
    best_idx : int
        Index of the best-fit candidate.
    save_filename : str
        Base filename (without extension) for the saved figure.
    result_dir : str
        Directory the returned figure path is anchored to.

    Returns
    -------
    fp_fig : str
        Filepath of the saved ``.png`` figure.
    """
    fig, ax = plt.subplots(1, figsize=(5.5, 4), constrained_layout=True)
    ax.plot(candidates, scores, "o-", color=spinna.NN_COLORS[0])
    ax.plot(
        candidates[best_idx],
        scores[best_idx],
        "o",
        color=spinna.NN_COLORS[2],
        markersize=10,
        label=f"best: {candidates[best_idx]:.2f} nm",
    )
    ax.set_xlabel("Labeling uncertainty (nm)")
    ax.set_ylabel("Kolmogorov-Smirnov score")
    ax.set_title(f"Labeling uncertainty screening: {target}")
    ax.legend()
    fname = f"{save_filename}_labelunc_scan_{target}"
    for ext in ["png", "svg"]:
        fig.savefig(f"{fname}.{ext}")
    plt.close(fig)
    return os.path.join(
        result_dir, f"{save_filename}_labelunc_scan_{target}.png"
    )


def screen_label_uncertainty(
    structures,
    label_unc,
    le,
    granularity,
    exp_data,
    mask_dict,
    width,
    height,
    depth,
    random_rot_mode,
    sim_repeats,
    asynch,
    result_dir,
    save_filename,
    fitting_mode="coarse-to-fine",
):
    """Screen a range of labeling uncertainties, one target at a time.

    For every molecular target whose ``label_unc`` entry lists more than
    one candidate, each candidate is scored by fitting a target-only
    sub-model to the experimental nearest-neighbour distances. This
    mirrors picasso's private ``spinna._fit_label_unc_for_target`` but
    retains the per-candidate scores so a screening curve can be plotted.
    The candidate with the lowest Kolmogorov-Smirnov score is selected.

    Parameters
    ----------
    structures : list of spinna.Structure
        The SPINNA model.
    label_unc : dict
        Maps each target name to a list of candidate labeling
        uncertainties in nm. A single-element list fixes that target's
        value (no screening).
    le : dict
        Labeling efficiency per target (0-1).
    granularity : int
        SPINNA granularity, see ``spinna.generate_N_structures``.
    exp_data : dict
        Experimental coordinates (nm) per target.
    mask_dict : dict or None
        Mask dictionary, see ``spinna.StructureMixer``.
    width, height, depth : float or None
        ROI dimensions in nm (``depth`` is None for 2D).
    random_rot_mode : {"2D", "3D"} or None
        Molecule rotation mode.
    sim_repeats : int
        Number of simulation repeats per candidate (``N_sim``).
    asynch : bool
        Whether picasso uses multiprocessing during fitting.
    result_dir : str
        Directory the returned figure paths are anchored to.
    save_filename : str
        Base filename (without extension) for saved figures/CSVs.
    fitting_mode : {"coarse-to-fine", "bayesian", "brute-force"}, optional
        Stoichiometry fitting mode forwarded to picasso. Default is
        "coarse-to-fine".

    Returns
    -------
    best_label_unc : dict
        Best-fit labeling uncertainty (float, nm) per target.
    scan : dict
        Maps each screened target to a dict with ``"candidates"`` and
        ``"scores"`` lists.
    fp_figs : list of str
        Filepaths of the saved screening figures (one per screened
        target).
    """
    targets = get_internal("_targets_from_structures")(structures)
    # nn_counts keys for all target pairs (reset per candidate fit)
    nn_keys = [
        f"{t1}-{t2}" for i, t1 in enumerate(targets) for t2 in targets[i:]
    ]
    # starting scalar for every target (first candidate), used for the
    # targets that are not currently being screened
    label_unc_start = {t: float(label_unc[t][0]) for t in targets}

    best_label_unc = {}
    scan = {}
    fp_figs = []
    for target in targets:
        candidates = [float(c) for c in label_unc[target]]
        if len(candidates) == 1:
            best_label_unc[target] = candidates[0]
            continue

        # only this target's monomers matter for its self-NND scan
        target_model = [s for s in structures if s.targets == [target]]
        if not target_model:
            logger.warning(
                f"No monomer structure found for target {target}; cannot "
                "screen its labeling uncertainty. Using the first candidate."
            )
            best_label_unc[target] = candidates[0]
            continue

        nn_counts = {key: 0 for key in nn_keys}
        nn_counts[f"{target}-{target}"] = 1

        scores = []
        for k, candidate in enumerate(candidates):
            label_unc_input = dict(label_unc_start)
            label_unc_input[target] = candidate
            score = spinna.compare_models_given_label_unc(
                models=[target_model],
                exp_data=exp_data,
                granularity=granularity,
                label_unc=label_unc_input,
                le=le,
                mask_dict=mask_dict,
                width=width,
                height=height,
                depth=depth,
                random_rot_mode=random_rot_mode,
                nn_counts=nn_counts,
                N_sim=sim_repeats,
                parallel=asynch,  # picasso 1.0: asynch -> parallel
                # empty savedir: the target sub-model here has a single
                # candidate, which crashes picasso's fit_stoichiometry
                # CSV-save branch (2-D N_structures vs 1-D props).
                savedir="",
                progress_title=(
                    f"Screening label_unc for {target}: "
                    f"{candidate:.2f} nm ({k + 1}/{len(candidates)})"
                ),
                fitting_mode=fitting_mode,
            )[0]
            scores.append(float(score))

        best_idx = int(np.argmin(scores))
        best_label_unc[target] = candidates[best_idx]
        scan[target] = {"candidates": candidates, "scores": scores}
        fp_figs.append(
            _plot_label_unc_scan(
                target=target,
                candidates=candidates,
                scores=scores,
                best_idx=best_idx,
                save_filename=save_filename,
                result_dir=result_dir,
            )
        )

    return best_label_unc, scan, fp_figs


def single_spinna_run(
    structures,
    label_unc,
    le,
    mask_dict,
    width,
    height,
    depth,
    random_rot_mode,
    exp_data,
    sim_repeats,
    NND_bin,
    NND_maxdist,
    N_structures,
    save_filename,
    asynch,
    targets,
    apply_mask,
    nn_plotted,
    result_dir,
    n_simulated,
    bootstrap=False,
):
    """Run a single SPINNA simulation with directly-given parameters.

    The implementation is adapted from the SPINNA batch analysis
    (``picasso.__main__._spinna_batch_analysis``) for a single run.

    Parameters
    ----------
    structures, label_unc, le, mask_dict, width, height, depth, \
            random_rot_mode, exp_data, sim_repeats, NND_bin, NND_maxdist, \
            N_structures, save_filename, asynch, targets, apply_mask, \
            nn_plotted, result_dir, n_simulated
        The SPINNA simulation parameters (see
        ``picasso.spinna.StructureMixer`` and the batch analysis).
    bootstrap : bool, optional
        Whether to bootstrap the standard error of the means. Default is
        False.

    Returns
    -------
    spinna_result : dict
        The results of the SPINNA run.
    fp_fig : list of str
        Filepaths of the NND figures.
    """
    mixer = spinna.StructureMixer(
        structures=structures,
        label_unc=label_unc,
        le=le,
        mask_dict=mask_dict,
        width=width,
        height=height,
        depth=depth,
        random_rot_mode=random_rot_mode,
    )

    # set up and run fitting
    opt_props, score = spinna.SPINNA(
        mixer=mixer,
        gt_coords=exp_data,
        N_sim=sim_repeats,
    ).fit_stoichiometry(
        N_structures,
        save=f"{save_filename}_fit_scores.csv",
        parallel=asynch,  # picasso 1.0: asynch -> parallel
        bootstrap=bootstrap,
    )

    # save the results
    results = {}
    results["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["File location of structures"] = save_filename
    results["Molecular targets"] = targets
    results["Labelling efficiency (%)"] = [
        le[target] * 100 for target in targets
    ]
    results["Label uncertainty (nm)"] = list(label_unc.values())
    results["Rotation mode"] = random_rot_mode
    results["AICc fitting score"] = score
    results["Fitted structures names"] = list(N_structures.keys())
    if isinstance(opt_props, tuple):
        props_mean, props_std = opt_props
        results["Modified Kolmogorov-Smirnov score +/- s.d."] = score
        results["Fitted proportions of structures"] = ", ".join(
            [
                f"{props_mean[i]:.2f} +/- {props_std[i]:.2f}%"
                for i in range(len(props_mean))
            ]
        )
        results["props"] = props_mean
        results["props_std"] = props_std
    else:
        results["Modified Kolmogorov-Smirnov score"] = score
        results["Fitted proportions of structures"] = opt_props
        results["props"] = opt_props
        results["props_std"] = [0] * len(opt_props)
    results["NND bin size (nm)"] = NND_bin
    results["NND max distance (nm)"] = NND_maxdist

    # relative proportions of structures for each target
    if len(targets) > 1:
        for target in targets:
            if isinstance(opt_props, tuple):
                rel_props = mixer.convert_props_for_target(
                    opt_props[0],
                    target,
                    n_simulated,
                )
                # rel_props_sd = mixer.convert_props_for_target(
                #     opt_props[1], target, n_simulated,
                # )
            else:
                rel_props = mixer.convert_props_for_target(
                    opt_props,
                    target,
                    n_simulated,
                )
            idx_valid = np.where(rel_props != np.inf)[0]
            value = ", ".join(
                [
                    f"{structures[i].title}: {rel_props[i]:.2f}%"
                    for i in idx_valid
                ]
            )
            results[f"Relative proportions of {target} in"] = value

    # save .txt with summary of the results
    with open(f"{save_filename}_fit_summary.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    # print(f"Results saved to {save_filename}_fit_summary.txt")

    # plot and save the NND plots
    fp_fig = plot_spinna_nnd(
        mixer=mixer,
        targets=targets,
        exp_data=exp_data,
        opt_props=opt_props,
        n_simulated=n_simulated,
        sim_repeats=sim_repeats,
        NND_bin=NND_bin,
        NND_maxdist=NND_maxdist,
        nn_plotted=nn_plotted,
        save_filename=save_filename,
        result_dir=result_dir,
    )

    return results, fp_fig


def plot_spinna_nnd(
    mixer,
    targets,
    exp_data,
    opt_props,
    n_simulated,
    sim_repeats,
    NND_bin,
    NND_maxdist,
    nn_plotted,
    save_filename,
    result_dir,
):
    """Plot simulated-vs-experimental NND histograms for a fitted mixer.

    Extracted from :func:`single_spinna_run` so it can be reused directly with
    the ``StructureMixer`` and proportions returned by ``spinna.fit_le``.

    Parameters
    ----------
    mixer : spinna.StructureMixer
        The fitted mixer.
    targets : list of str
        Molecular target names; figures are produced for each (duplicated)
        target pair, in the order given by
        ``mixer.get_neighbor_idx(duplicate=True)``.
    exp_data : dict
        Experimental coordinates per target.
    opt_props : np.ndarray or tuple
        Fitted structure proportions (a tuple if bootstrapped; the mean is
        used).
    n_simulated : dict
        Number of simulated molecules per target.
    sim_repeats : int
        Number of simulation repeats.
    NND_bin : float
        Histogram bin size in nm.
    NND_maxdist : float
        Maximum distance shown in nm.
    nn_plotted : int
        Number of nearest neighbours to plot.
    save_filename : str
        Base filename (without extension) for the saved figures.
    result_dir : str
        Directory the returned figure paths are anchored to.

    Returns
    -------
    fp_fig : list of str
        Filepaths of the saved NND ``.png`` figures.
    """
    nn_counts = {}
    for i, t1 in enumerate(targets):
        for t2 in targets[i:]:
            nn_counts[f"{t1}-{t2}"] = nn_plotted
    mixer.nn_counts = nn_counts
    # dist_sim = spinna.get_NN_dist_repeated(
    # # dist_sim = get_NN_dist_repeated(
    #     opt_N_str, sim_repeats, mixer,
    #     duplicate=True
    # )
    n_total = sum(n_simulated.values())
    if isinstance(opt_props, tuple):
        opt_prop_vals = opt_props[0]
    else:
        opt_prop_vals = opt_props
    dist_sim = spinna.get_NN_dist_simulated(
        mixer.convert_props_to_counts(opt_prop_vals, n_total),
        sim_repeats,
        mixer,
        duplicate=True,
    )
    fp_fig = []
    for i, (t1, t2, _) in enumerate(mixer.get_neighbor_idx(duplicate=True)):
        # fig, ax = plot_NN(
        fig, ax = spinna.plot_NN(
            dist=dist_sim[i],
            mode="plot",
            show_legend=False,
            return_fig=True,
            figsize=(5.5, 4),
            alpha=1.0,
            binsize=NND_bin,
            xlim=[0, NND_maxdist],
            title=f"Nearest Neighbors Distances: {t1} -> {t2}",
        )
        exp1 = exp_data[t1]
        exp2 = exp_data[t2]
        # fig, ax = plot_NN(
        fig, ax = spinna.plot_NN(
            data1=exp1,
            data2=exp2,
            n_neighbors=nn_plotted,
            show_legend=False,
            fig=fig,
            ax=ax,
            mode="hist",
            return_fig=True,
            binsize=NND_bin,
            xlim=[0, NND_maxdist],
            title=f"Nearest Neighbors Distances: {t1} -> {t2}",
            savefig=[
                f"{save_filename}_NND_{t1}_{t2}.{_}" for _ in ["png", "svg"]
            ],
        )
        fp_fig.append(
            os.path.join(result_dir, f"{save_filename}_NND_{t1}_{t2}.png")
        )

    return fp_fig


def single_spinna_fit_le_run(
    target_a,
    target_b,
    exp_data,
    granularity,
    label_unc,
    distances,
    mask_dict,
    width,
    height,
    depth,
    random_rot_mode,
    sim_repeats,
    asynch,
    NND_bin,
    NND_maxdist,
    nn_plotted,
    n_simulated,
    result_dir,
    save_filename,
    fitting_mode="coarse-to-fine",
):
    """Fit labeling efficiency and screen the pair distance of two targets.

    Thin wrapper around picasso's ``spinna.fit_le``. It builds the
    monomer-A / monomer-B / heterodimer model family internally, screens
    the heterodimer separation over ``distances`` (and, per target, the
    labeling uncertainty over ``label_unc``), forces labeling efficiency
    to 100% during the fit and recovers it from the fitted proportions.
    The ``structures`` and ``labeling_efficiency`` inputs of the SPINNA
    module are therefore not used in this mode.

    Parameters
    ----------
    target_a, target_b : str
        The two molecular target names; both must be keys in
        ``exp_data``.
    exp_data : dict
        Experimental coordinates (nm) per target.
    granularity : int
        SPINNA granularity, see ``spinna.generate_N_structures``.
    label_unc : dict
        Maps each target to a list of candidate labeling uncertainties in
        nm. A single-element list fixes that target's value.
    distances : list of float
        Candidate heterodimer separations in nm to screen.
    mask_dict : dict or None
        Mask dictionary, see ``spinna.StructureMixer``.
    width, height, depth : float or None
        ROI dimensions in nm (``depth`` is None for 2D).
    random_rot_mode : {"2D", "3D"} or None
        Molecule rotation mode.
    sim_repeats : int
        Number of simulation repeats (``N_sim``).
    asynch : bool
        Whether picasso uses multiprocessing during fitting.
    NND_bin : float
        Histogram bin size in nm.
    NND_maxdist : float
        Maximum distance shown in nm.
    nn_plotted : int
        Number of nearest neighbours to plot.
    n_simulated : dict
        Number of molecules per target used when plotting the fitted
        NND histograms.
    result_dir : str
        Directory the returned figure paths are anchored to.
    save_filename : str
        Base filename (without extension) for saved figures/summary.
    fitting_mode : {"coarse-to-fine", "bayesian", "brute-force"}, optional
        Stoichiometry fitting mode forwarded to picasso. Default is
        "coarse-to-fine".

    Returns
    -------
    results : dict
        Human-readable summary of the fit.
    fp_fig : list of str
        Filepaths of the saved NND figures.
    le_values : dict
        Fitted labeling efficiency [%] per target.
    fitted_label_unc : dict
        Fitted labeling uncertainty [nm] per target.
    best_distance : float
        Best-fit heterodimer separation in nm.
    best_score : float
        Kolmogorov-Smirnov score of the best fit.
    """
    (
        le_values,
        fitted_label_unc,
        best_distance,
        best_score,
        best_props,
        best_mixer,
    ) = spinna.fit_le(
        target_a=target_a,
        target_b=target_b,
        exp_data=exp_data,
        granularity=granularity,
        label_unc=label_unc,
        distances=distances,
        N_sim=sim_repeats,
        mask_dict=mask_dict,
        width=width,
        height=height,
        depth=depth,
        random_rot_mode=random_rot_mode,
        parallel=asynch,  # picasso 1.0: asynch -> parallel
        # empty savedir: avoid picasso's fit_stoichiometry crash on the
        # single-candidate target sub-model used when screening labeling
        # uncertainty (see labeling_efficiency_analysis for details).
        savedir="",
        fitting_mode=fitting_mode,
    )

    targets = [target_a, target_b]
    results = {}
    results["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["File location of structures"] = save_filename
    results["Molecular targets"] = targets
    results["Fitted labeling efficiency (%)"] = {
        target: le_values[target] for target in targets
    }
    results["Fitted label uncertainty (nm)"] = {
        target: fitted_label_unc[target] for target in targets
    }
    results["Best pair distance (nm)"] = best_distance
    results["Rotation mode"] = random_rot_mode
    results["Modified Kolmogorov-Smirnov score"] = best_score
    results["Fitted structures names"] = best_mixer.get_structure_names()
    results["Fitted proportions of structures"] = best_props
    results["NND bin size (nm)"] = NND_bin
    results["NND max distance (nm)"] = NND_maxdist

    # save .txt with summary of the results
    with open(f"{save_filename}_fit_le_summary.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    # plot and save the NND plots for the best-fit mixer
    fp_fig = plot_spinna_nnd(
        mixer=best_mixer,
        targets=targets,
        exp_data=exp_data,
        opt_props=best_props,
        n_simulated=n_simulated,
        sim_repeats=sim_repeats,
        NND_bin=NND_bin,
        NND_maxdist=NND_maxdist,
        nn_plotted=nn_plotted,
        save_filename=save_filename,
        result_dir=result_dir,
    )

    return (
        results,
        fp_fig,
        le_values,
        fitted_label_unc,
        best_distance,
        best_score,
    )


def load_structures_from_dict(structure_dict):
    """Loads structures (SingleStructure's) from dict with format as
    those saved in .yaml files.

    Parameters
    ----------
    structure_dict : list of dict
        structure description as dict.

    Returns
    -------
    structures : list of SingleStructure's
        List of structures loaded from the file.
    targets : list of strs
        List of all unique molecular targets in the structures.
    """
    if "Structure title" not in structure_dict[0].keys():
        raise TypeError(
            "Incorrect file. Please choose a file that was created"
            " that was created with Picasso SPINNA."
        )
    # continue if the correct file is loaded
    structures = []
    targets = []
    for m_info in structure_dict:
        structure = spinna.Structure(m_info["Structure title"])
        for target in m_info["Molecular targets"]:
            x = m_info[f"{target}_x"]
            y = m_info[f"{target}_y"]
            z = m_info[f"{target}_z"]
            structure.define_coordinates(target, x, y, z)
            if target not in targets:
                targets.append(target)
        structures.append(structure)
    return structures, targets
