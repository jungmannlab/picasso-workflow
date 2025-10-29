"""
Iterative RSSO-based Drift Correction

This module implements iterative Redundant Spot Shift Overrepresentation (RSSO)
drift correction where each frame is compared against the whole dataset to compute
total drift for that frame. The process is repeated iteratively with the undrifted
dataset to improve accuracy.

Key features:
- Iterative refinement for improved drift estimation accuracy
- Uncertainty analysis and confidence evaluation
- Adaptive windowing for low-confidence frames
- Outlier detection and filtering
- Performance optimization with subsampling and Numba acceleration
- Memory-efficient chunked processing

Author: Generated for picasso-workflow
"""

import numpy as np

# import logging
from loguru import logger
import multiprocessing as mp
import time
import os

# # logger = logging.getLogger(__name__)

# # Configure OpenMP BEFORE any potential OpenMP initialization
# # This prevents fork() conflicts when using multiprocessing
# # Must be set at module level before numerical libraries initialize
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Try to import Numba for acceleration
try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print(
        "Warning: Numba not available. RSSO computations will use standard NumPy (slower)."
    )


# ==============================================================================
# Multiprocessing configuration
# ==============================================================================


def _configure_openmp_for_multiprocessing():
    """Configure OpenMP environment to avoid conflicts with multiprocessing

    DEPRECATED: This function is kept for documentation purposes only.
    OpenMP configuration is now done at module import time (see module-level
    code after imports) to ensure environment variables are set before any
    numerical libraries initialize OpenMP threads.

    This fixes the error: "Terminating: fork() called from a process already
    using GNU OpenMP, this is unsafe."

    The issue occurs when NumPy/SciPy compiled with OpenMP are used in
    combination with multiprocessing fork(). Setting these environment
    variables disables OpenMP threading in worker processes.

    The configuration is now applied at module level to ensure it takes
    effect before any OpenMP initialization, which is critical for
    SLURM/cluster environments where each MPI rank uses multiprocessing.
    """
    # Configuration now done at module level - this function is a no-op
    pass


def _setup_multiprocessing_context():
    """Setup multiprocessing context to avoid OpenMP conflicts"""
    try:
        # Try to set multiprocessing start method to 'spawn' to avoid fork() issues
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        return mp.get_context("spawn")
    except RuntimeError:
        # If already set or unavailable, use default context
        try:
            return mp.get_context("spawn")
        except ValueError:
            # 'spawn' not available, fall back to default
            return mp.get_context()


# Global variables to hold pre-built data structures in worker processes
_WORKER_KDTREE = None
_WORKER_KDTREE_SHM = None
_WORKER_FRAMES = None  # Reference frame numbers for temporal filtering
_WORKER_FRAMES_SHM = None  # Shared memory reference for frame array


def _worker_init_kdtree(ref_coords):
    """Initialize worker process by building cKDTree once

    This function is called once per worker process during pool initialization.
    Building the cKDTree once per worker (rather than passing it with each task)
    significantly reduces data transfer overhead.

    Args:
        ref_coords : ndarray
            Reference coordinates (N x 2) array of [x, y] positions
    """
    global _WORKER_KDTREE
    from scipy.spatial import cKDTree

    _WORKER_KDTREE = cKDTree(ref_coords)
    logger.debug(f"Worker initialized: built cKDTree with {_WORKER_KDTREE.n} points")


def _init_worker_from_shared_memory(
    shm_name, kdtree_size, frame_shm_name=None, frame_array_len=None, frame_dtype=None
):
    """Initialize worker by loading cKDTree and frame array from shared memory

    This function provides zero-copy access to a pre-built cKDTree and reference
    frame array by deserializing them from shared memory. This is faster than
    building the tree from coordinates and uses less memory than per-worker copies.

    Args:
        shm_name : str
            Name of the cKDTree shared memory segment
        kdtree_size : int
            Size of the serialized cKDTree in bytes
        frame_shm_name : str, optional
            Name of the frame array shared memory segment
        frame_array_len : int, optional
            Number of elements in frame array
        frame_dtype : dtype, optional
            Data type of frame array (typically np.int32)
    """
    global _WORKER_KDTREE, _WORKER_KDTREE_SHM, _WORKER_FRAMES, _WORKER_FRAMES_SHM
    from multiprocessing import shared_memory
    import pickle

    # Attach to existing cKDTree shared memory
    shm = shared_memory.SharedMemory(name=shm_name)

    # Deserialize cKDTree from shared memory
    kdtree_bytes = bytes(shm.buf[:kdtree_size])
    _WORKER_KDTREE = pickle.loads(kdtree_bytes)

    # Store shared memory reference for potential cleanup
    _WORKER_KDTREE_SHM = shm

    logger.debug(
        f"Worker initialized cKDTree from shared memory: {_WORKER_KDTREE.n} points"
    )

    # Load frame array if provided (for temporal filtering)
    if frame_shm_name is not None and frame_array_len is not None:
        frame_shm = shared_memory.SharedMemory(name=frame_shm_name)
        frame_bytes = bytes(frame_shm.buf[: frame_array_len * np.dtype(frame_dtype).itemsize])
        _WORKER_FRAMES = np.frombuffer(frame_bytes, dtype=frame_dtype)
        _WORKER_FRAMES_SHM = frame_shm

        logger.debug(
            f"Worker initialized frame array from shared memory: {len(_WORKER_FRAMES):,} frames"
        )


def _create_shared_memory_kdtree(reference_coords):
    """Serialize cKDTree to shared memory for zero-copy worker access

    Creates a cKDTree from reference coordinates, serializes it using pickle,
    and stores it in shared memory. Workers can then deserialize from the
    shared memory without requiring per-task or per-worker data transfer.

    Args:
        reference_coords : ndarray
            Reference coordinates (N x 2) array of [x, y] positions

    Returns:
        tuple : (SharedMemory, int)
            Shared memory object and size in bytes of the serialized cKDTree
    """
    from scipy.spatial import cKDTree
    from multiprocessing import shared_memory
    import pickle

    # Build tree and serialize with highest protocol for efficiency
    kdtree = cKDTree(reference_coords)
    kdtree_bytes = pickle.dumps(kdtree, protocol=pickle.HIGHEST_PROTOCOL)
    kdtree_size = len(kdtree_bytes)

    # Create shared memory segment
    shm = shared_memory.SharedMemory(create=True, size=kdtree_size)
    shm.buf[:kdtree_size] = kdtree_bytes

    logger.debug(
        f"Created shared memory cKDTree: {kdtree.n} points, "
        f"{kdtree_size / 1024 / 1024:.2f} MB"
    )

    return shm, kdtree_size


def _create_shared_memory_frame_array(frame_array):
    """Serialize frame array to shared memory for zero-copy worker access

    Stores frame numbers in shared memory so workers can filter pairs by
    temporal proximity without serializing the array with each task.

    Args:
        frame_array : ndarray
            Frame numbers (N,) array of integers

    Returns:
        tuple : (SharedMemory, int, dtype)
            Shared memory object, array size, and dtype for reconstruction
    """
    from multiprocessing import shared_memory

    # Convert to int32 for memory efficiency
    frame_array_int32 = frame_array.astype(np.int32)
    array_bytes = frame_array_int32.tobytes()
    array_size = len(array_bytes)

    # Create shared memory segment
    shm = shared_memory.SharedMemory(create=True, size=array_size)
    shm.buf[:array_size] = array_bytes

    logger.debug(
        f"Created shared memory frame array: {len(frame_array):,} frames, "
        f"{array_size / 1024 / 1024:.2f} MB"
    )

    return shm, len(frame_array), np.int32


# ==============================================================================
# Profiling and timing utilities
# ==============================================================================


class _Timer:
    """Context manager for timing code blocks and operations

    Usage:
        with _Timer("operation_name") as timer:
            # code to time
        elapsed = timer.elapsed  # seconds
    """

    def __init__(self, name="operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        return False


def _format_time(seconds):
    """Format seconds into human-readable string

    Args:
        seconds : float
            Time in seconds

    Returns:
        str : Formatted string like "2.34 sec" or "3.45 min" or "1.23 hr"
    """
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} hr"


def _log_performance_summary(
    iteration,
    total_frames,
    timings,
    method_name,
    n_processes,
):
    """Log detailed performance summary for an iteration with comprehensive breakdown

    Args:
        iteration : int
            Iteration number (1-indexed)
        total_frames : int
            Number of frames processed
        timings : dict
            Dictionary with timing information including all fine-grained metrics
        method_name : str
            KDTree sharing method name
        n_processes : int
            Number of worker processes
    """
    total_time = timings["total"]

    def pct(t):
        """Calculate percentage of total time"""
        return (t / total_time * 100) if total_time > 0 else 0

    # Extract all timing metrics
    # Setup phase
    reference_creation = timings.get("reference_creation", 0.0)
    kdtree_creation = timings.get("kdtree_creation", 0.0)
    kdtree_serialization = timings.get("kdtree_serialization", 0.0)
    setup_total = reference_creation + kdtree_creation + kdtree_serialization

    # Multiprocessing phase
    pool_creation = timings.get("pool_creation", 0.0)
    worker_initialization = timings.get("worker_initialization", 0.0)
    pool_map_total = timings.get("pool_map_total", 0.0)
    multiprocessing_overhead = timings.get("multiprocessing_overhead", 0.0)
    mp_total = pool_creation + worker_initialization + pool_map_total

    # Processing phase (frame-level operations)
    frame_grouping = timings.get("frame_grouping", 0.0)
    chunk_data_preparation = timings.get("chunk_data_preparation", 0.0)
    worker_computation = timings.get("worker_computation", 0.0)
    result_collection = timings.get("result_collection", 0.0)
    processing_total = frame_grouping + chunk_data_preparation + result_collection

    # Post-processing phase (localization-level operations)
    array_copy = timings.get("array_copy", 0.0)
    frame_pregrouping = timings.get("frame_pregrouping", 0.0)
    windowing_outliers = timings.get("windowing_outliers", 0.0)
    array_operations = timings.get("array_operations", 0.0)
    frame_corrections = timings.get("frame_corrections", 0.0)
    postproc_total = array_copy + frame_pregrouping + windowing_outliers + array_operations + frame_corrections

    # Finalization phase
    convergence_check = timings.get("convergence_check", 0.0)
    history_storage = timings.get("history_storage", 0.0)
    finalization_total = convergence_check + history_storage

    # Calculate throughput
    frame_processing_time = timings.get("frame_processing", 0.0)
    frames_per_sec = total_frames / frame_processing_time if frame_processing_time > 0 else 0
    time_per_frame = frame_processing_time / total_frames if total_frames > 0 else 0

    # Log comprehensive summary
    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"  Iteration {iteration} Performance Summary - {method_name} ({n_processes} workers)")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {_format_time(total_time)}")
    logger.info("")

    # Setup phase
    logger.info(f"┌─ SETUP: {_format_time(setup_total)} ({pct(setup_total):.1f}%)")
    if reference_creation > 0:
        logger.info(f"│  ├─ Reference creation: {_format_time(reference_creation)} ({pct(reference_creation):.1f}%)")
    logger.info(f"│  ├─ KDTree creation: {_format_time(kdtree_creation)} ({pct(kdtree_creation):.1f}%)")
    if kdtree_serialization > 0:
        logger.info(f"│  └─ KDTree serialization: {_format_time(kdtree_serialization)} ({pct(kdtree_serialization):.1f}%)")
    logger.info("")

    # Multiprocessing phase
    if mp_total > 0:
        logger.info(f"├─ MULTIPROCESSING: {_format_time(mp_total)} ({pct(mp_total):.1f}%)")
        if pool_creation > 0:
            logger.info(f"│  ├─ Pool creation: {_format_time(pool_creation)} ({pct(pool_creation):.1f}%)")
        if worker_initialization > 0:
            logger.info(f"│  ├─ Worker initialization: {_format_time(worker_initialization)} ({pct(worker_initialization):.1f}%)")
        if pool_map_total > 0:
            logger.info(f"│  ├─ Pool.map total: {_format_time(pool_map_total)} ({pct(pool_map_total):.1f}%)")
            if worker_computation > 0:
                logger.info(f"│  │  ├─ Worker computation: {_format_time(worker_computation)} ({pct(worker_computation):.1f}%)")
            if multiprocessing_overhead > 0:
                mp_overhead_pct = (multiprocessing_overhead / pool_map_total * 100) if pool_map_total > 0 else 0
                logger.info(f"│  │  └─ MP overhead: {_format_time(multiprocessing_overhead)} ({mp_overhead_pct:.1f}% of pool.map)")
        logger.info("")

    # Processing phase
    logger.info(f"├─ PROCESSING: {_format_time(processing_total)} ({pct(processing_total):.1f}%)")
    if frame_grouping > 0:
        logger.info(f"│  ├─ Frame grouping: {_format_time(frame_grouping)} ({pct(frame_grouping):.1f}%)")
    if chunk_data_preparation > 0:
        logger.info(f"│  ├─ Chunk data prep: {_format_time(chunk_data_preparation)} ({pct(chunk_data_preparation):.1f}%)")
    if result_collection > 0:
        logger.info(f"│  └─ Result collection: {_format_time(result_collection)} ({pct(result_collection):.1f}%)")
    logger.info(f"│     └─ Total frames: {total_frames} @ {frames_per_sec:.2f} frames/sec")
    logger.info("")

    # Post-processing phase
    logger.info(f"├─ POST-PROCESSING: {_format_time(postproc_total)} ({pct(postproc_total):.1f}%)")
    if array_copy > 0:
        logger.info(f"│  ├─ Array copy: {_format_time(array_copy)} ({pct(array_copy):.1f}%)")
    if frame_pregrouping > 0:
        logger.info(f"│  ├─ Frame pre-grouping: {_format_time(frame_pregrouping)} ({pct(frame_pregrouping):.1f}%)")
    if windowing_outliers > 0:
        logger.info(f"│  ├─ Windowing/outliers: {_format_time(windowing_outliers)} ({pct(windowing_outliers):.1f}%)")
    if array_operations > 0:
        logger.info(f"│  ├─ Array operations: {_format_time(array_operations)} ({pct(array_operations):.1f}%)")
    if frame_corrections > 0:
        logger.info(f"│  └─ Frame corrections: {_format_time(frame_corrections)} ({pct(frame_corrections):.1f}%)")
    logger.info("")

    # Finalization phase
    logger.info(f"└─ FINALIZATION: {_format_time(finalization_total)} ({pct(finalization_total):.1f}%)")
    if convergence_check > 0:
        logger.info(f"   ├─ Convergence check: {_format_time(convergence_check)} ({pct(convergence_check):.1f}%)")
    if history_storage > 0:
        logger.info(f"   └─ History storage: {_format_time(history_storage)} ({pct(history_storage):.1f}%)")

    # Sanity check: sum of phases vs total
    phase_sum = setup_total + mp_total + processing_total + postproc_total + finalization_total
    unaccounted = total_time - phase_sum
    if abs(unaccounted) > 0.1:  # More than 0.1 second unaccounted
        logger.info("")
        logger.info(f"⚠️  Unaccounted time: {_format_time(abs(unaccounted))} ({pct(abs(unaccounted)):.1f}%)")

    logger.info("")
    logger.info(f"{'='*70}")
    logger.info("")

    # Log chunk timing details if available
    if "chunk_times" in timings and timings["chunk_times"]:
        chunk_times = timings["chunk_times"]
        avg_chunk = np.mean(chunk_times)
        min_chunk = np.min(chunk_times)
        max_chunk = np.max(chunk_times)
        logger.debug(
            f"Chunk timing: avg={_format_time(avg_chunk)}, "
            f"min={_format_time(min_chunk)}, max={_format_time(max_chunk)}"
        )


# ==============================================================================
# Numba-optimized RSSO computation functions
# ==============================================================================

if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _compute_pairwise_shifts_numba(
        i_x, i_y, j_x, j_y, max_shift=None, i_frames=None, j_frames=None, ton_exclusion=0
    ):
        """Numba-optimized pairwise shift computation with temporal filtering

        Computes shifts from i to j: j - i (matches standard implementation)
        Only includes pairs within max_shift distance (like standard KDTree approach)
        Optionally excludes pairs from temporally close frames (within ±2×ton)

        Note: Uses parallel=True with fixed-size arrays for optimal performance

        Args:
            i_x, i_y : ndarray
                First set of coordinates (reference in standard call)
            j_x, j_y : ndarray
                Second set of coordinates (frame in standard call)
            max_shift : float, optional
                Maximum distance to consider pairs (matches standard implementation)
            i_frames : ndarray, optional
                Frame numbers for i coordinates (for temporal filtering)
            j_frames : ndarray, optional
                Frame numbers for j coordinates (for temporal filtering)
            ton_exclusion : int, default 0
                Exclude pairs from frames within ±2×ton (temporal filtering)

        Returns:
            shifts_x, shifts_y : ndarray
                Valid pairwise shift vectors from i to j
        """
        n_i = len(i_x)
        n_j = len(j_x)
        max_shift_sq = (
            max_shift * max_shift if max_shift is not None else np.inf
        )

        # Pre-allocate maximum possible size
        max_pairs = n_i * n_j
        all_shifts_x = np.empty(max_pairs, dtype=numba.float32)
        all_shifts_y = np.empty(max_pairs, dtype=numba.float32)
        valid_mask = np.zeros(max_pairs, dtype=numba.boolean)

        # Determine if temporal filtering is enabled
        use_temporal_filter = (
            ton_exclusion > 0 and i_frames is not None and j_frames is not None
        )
        temporal_threshold = 2 * ton_exclusion

        # Parallel computation over pairs
        for idx in numba.prange(
            max_pairs
        ):  # Parallel over all potential pairs
            i = idx // n_j  # Convert flat index to i,j coordinates
            j = idx % n_j

            # Temporal filtering: skip pairs from nearby frames
            if use_temporal_filter:
                frame_diff = abs(i_frames[i] - j_frames[j])
                if frame_diff <= temporal_threshold:
                    valid_mask[idx] = False
                    continue

            i_x_val = i_x[i]
            i_y_val = i_y[i]

            dx = j_x[j] - i_x_val
            dy = j_y[j] - i_y_val

            # Spatial filtering: check distance
            if max_shift is None or (dx * dx + dy * dy) <= max_shift_sq:
                all_shifts_x[idx] = dx
                all_shifts_y[idx] = dy
                valid_mask[idx] = True
            else:
                valid_mask[idx] = False

        # Count valid pairs sequentially (avoid race conditions)
        n_valid = 0
        for k in range(max_pairs):
            if valid_mask[k]:
                n_valid += 1

        # Create compact output arrays
        shifts_x = np.empty(n_valid, dtype=numba.float32)
        shifts_y = np.empty(n_valid, dtype=numba.float32)

        # Copy valid pairs sequentially
        valid_idx = 0
        for k in range(max_pairs):
            if valid_mask[k]:
                shifts_x[valid_idx] = all_shifts_x[k]
                shifts_y[valid_idx] = all_shifts_y[k]
                valid_idx += 1

        return shifts_x, shifts_y

    @numba.jit(
        nopython=True, parallel=False, cache=True
    )  # Sequential for thread safety in binning
    def _histogram2d_numba(shifts_x, shifts_y, x_edges, y_edges):
        """Numba-optimized 2D histogram binning (Phase 2)

        Args:
            shifts_x, shifts_y : ndarray
                Shift vectors to bin
            x_edges, y_edges : ndarray
                Bin edges for histogram

        Returns:
            hist : ndarray
                2D histogram counts
        """
        nx_bins = len(x_edges) - 1
        ny_bins = len(y_edges) - 1
        hist = np.zeros((nx_bins, ny_bins), dtype=numba.int32)

        n_points = len(shifts_x)

        # Custom binning loop
        for i in range(n_points):
            x_val = shifts_x[i]
            y_val = shifts_y[i]

            # Find bins using binary search equivalent
            x_bin = -1
            y_bin = -1

            # Simple linear search for bin (could be optimized further)
            for j in range(nx_bins):
                if x_edges[j] <= x_val < x_edges[j + 1]:
                    x_bin = j
                    break

            for j in range(ny_bins):
                if y_edges[j] <= y_val < y_edges[j + 1]:
                    y_bin = j
                    break

            # Increment histogram bin if valid
            if 0 <= x_bin < nx_bins and 0 <= y_bin < ny_bins:
                hist[x_bin, y_bin] += 1

        return hist

    @numba.jit(nopython=True, cache=True)
    def _find_histogram_peak_numba(hist):
        """Numba-optimized peak finding with sub-pixel refinement

        Args:
            hist : ndarray
                2D histogram

        Returns:
            peak_x, peak_y : float
                Peak location with sub-pixel precision
            peak_value : int
                Peak histogram value
        """
        max_val = 0
        max_i = 0
        max_j = 0

        # Find maximum value and location
        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                if hist[i, j] > max_val:
                    max_val = hist[i, j]
                    max_i = i
                    max_j = j

        # Sub-pixel refinement using parabolic interpolation
        peak_x = numba.float32(max_i)
        peak_y = numba.float32(max_j)

        # Parabolic refinement in x-direction
        if 0 < max_i < hist.shape[0] - 1:
            left = hist[max_i - 1, max_j]
            center = hist[max_i, max_j]
            right = hist[max_i + 1, max_j]

            # Parabolic peak formula: offset = (left - right) / (2 * (left - 2*center + right))
            denominator = 2 * (left - 2 * center + right)
            if abs(denominator) > 1e-6:  # Avoid division by zero
                offset_x = (left - right) / denominator
                peak_x = max_i + offset_x

        # Parabolic refinement in y-direction
        if 0 < max_j < hist.shape[1] - 1:
            bottom = hist[max_i, max_j - 1]
            center = hist[max_i, max_j]
            top = hist[max_i, max_j + 1]

            denominator = 2 * (bottom - 2 * center + top)
            if abs(denominator) > 1e-6:
                offset_y = (bottom - top) / denominator
                peak_y = max_j + offset_y

        return peak_x, peak_y, max_val

else:
    # Fallback implementations when Numba is not available
    def _compute_pairwise_shifts_numba(
        i_x, i_y, j_x, j_y, max_shift=None, i_frames=None, j_frames=None, ton_exclusion=0
    ):
        """Fallback NumPy implementation with temporal filtering"""
        i_coords = np.column_stack([i_x, i_y])
        j_coords = np.column_stack([j_x, j_y])

        shifts_x = []
        shifts_y = []

        use_temporal_filter = (
            ton_exclusion > 0 and i_frames is not None and j_frames is not None
        )
        temporal_threshold = 2 * ton_exclusion

        for i, i_coord in enumerate(i_coords):
            # Calculate shift from i to j: j - i (matches standard)
            dx = j_coords[:, 0] - i_coord[0]
            dy = j_coords[:, 1] - i_coord[1]

            # Start with all pairs valid
            valid_mask = np.ones(len(dx), dtype=bool)

            # Temporal filtering: exclude pairs from nearby frames
            if use_temporal_filter:
                frame_diffs = np.abs(j_frames - i_frames[i])
                temporal_mask = frame_diffs > temporal_threshold
                valid_mask &= temporal_mask

            # Spatial filtering: exclude pairs beyond max_shift
            if max_shift is not None:
                distances_sq = dx * dx + dy * dy
                spatial_mask = distances_sq <= (max_shift * max_shift)
                valid_mask &= spatial_mask

            # Apply combined mask
            dx = dx[valid_mask]
            dy = dy[valid_mask]

            shifts_x.extend(dx)
            shifts_y.extend(dy)

        return (
            np.array(shifts_x, dtype=np.float32),
            np.array(shifts_y, dtype=np.float32),
        )

    def _histogram2d_numba(shifts_x, shifts_y, x_edges, y_edges):
        """Fallback NumPy implementation"""
        hist, _, _ = np.histogram2d(
            shifts_x, shifts_y, bins=[x_edges, y_edges]
        )
        return hist.astype(np.int32)

    def _find_histogram_peak_numba(hist):
        """Fallback NumPy implementation"""
        max_idx = np.unravel_index(hist.argmax(), hist.shape)
        return float(max_idx[0]), float(max_idx[1]), hist[max_idx]


# ==============================================================================
# Optimized frame correction application
# ==============================================================================


if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _apply_frame_corrections_numba(
        frame_shifts_x, frame_shifts_y, frame_index_map, pixelsize
    ):
        """Apply per-frame shifts to per-localization corrections using Numba

        This is much faster than NumPy fancy indexing for large arrays.
        Uses parallel processing for optimal performance.

        Args:
            frame_shifts_x : ndarray (float)
                Per-frame shifts in x (nm)
            frame_shifts_y : ndarray (float)
                Per-frame shifts in y (nm)
            frame_index_map : ndarray (int32)
                Maps each localization to its frame index
            pixelsize : float
                Pixel size for conversion from nm to pixels

        Returns:
            corrections_x, corrections_y : tuple of ndarray (float32)
                Per-localization corrections in pixels
        """
        n_locs = len(frame_index_map)
        corrections_x = np.empty(n_locs, dtype=np.float32)
        corrections_y = np.empty(n_locs, dtype=np.float32)

        # Parallel loop over all localizations
        for i in numba.prange(n_locs):
            frame_idx = frame_index_map[i]
            corrections_x[i] = frame_shifts_x[frame_idx] / pixelsize
            corrections_y[i] = frame_shifts_y[frame_idx] / pixelsize

        return corrections_x, corrections_y

else:

    def _apply_frame_corrections_numba(
        frame_shifts_x, frame_shifts_y, frame_index_map, pixelsize
    ):
        """Fallback NumPy implementation when Numba is not available"""
        corrections_x = frame_shifts_x[frame_index_map] / pixelsize
        corrections_y = frame_shifts_y[frame_index_map] / pixelsize
        return corrections_x.astype(np.float32), corrections_y.astype(np.float32)


# ==============================================================================
# Main RSSO shift computation
# ==============================================================================


def _compute_rsso_shift_numba_optimized(
    locs_i,
    locs_j,
    max_shift_pixels,
    enable_numba=True,
    plot_histogram=False,
    plot_dir=None,
    iteration=None,
    frame_number=None,
    ref_frames=None,
    frame_locs_frames=None,
    ton=0,
):
    """Numba-optimized RSSO shift computation with temporal filtering

    Args:
        locs_i : ndarray
            First set of localizations (reference in standard call)
        locs_j : ndarray
            Second set of localizations (frame in standard call)
        max_shift_pixels : float
            Maximum expected shift in pixels
        enable_numba : bool
            Whether to use Numba optimization
        plot_histogram : bool
            Whether to save 2D histogram plot
        plot_dir : str, optional
            Directory to save plots
        iteration : int, optional
            Iteration number for filename
        frame_number : int, optional
            Frame number for filename
        ref_frames : ndarray, optional
            Frame numbers for reference localizations (for temporal filtering)
        frame_locs_frames : ndarray, optional
            Frame numbers for frame localizations (for temporal filtering)
        ton : int, default 0
            Exclude pairs from frames within ±2×ton (temporal filtering)

    Returns:
        shift_x, shift_y : float
            Detected shift from locs_i to locs_j (same as standard)
        quality_metrics : dict
            Quality and uncertainty information
    """
    import time

    if len(locs_i) == 0 or len(locs_j) == 0:
        return None, None, {"success": False, "reason": "insufficient_data"}

    start_time = time.time()

    # Extract coordinates - match standard implementation naming
    # locs_i is reference, locs_j is frame in standard call pattern
    i_x = locs_i["x"].astype(np.float32)
    i_y = locs_i["y"].astype(np.float32)
    j_x = locs_j["x"].astype(np.float32)
    j_y = locs_j["y"].astype(np.float32)

    # Extract frame numbers if available for temporal filtering
    if ref_frames is not None:
        i_frames = ref_frames.astype(np.int32) if not isinstance(ref_frames, np.ndarray) else ref_frames.astype(np.int32)
    else:
        i_frames = None

    if frame_locs_frames is not None:
        j_frames = frame_locs_frames.astype(np.int32) if not isinstance(frame_locs_frames, np.ndarray) else frame_locs_frames.astype(np.int32)
    else:
        j_frames = None

    phase1_start = time.time()

    # Phase 1: Compute all pairwise shifts with temporal filtering (Numba optimized)
    # Standard calculation: dx = coord_j[0] - coord_i[0] (j - i)
    # Only consider pairs within max_shift distance (like standard KDTree approach)
    # Exclude pairs from temporally close frames (within ±2×ton)
    if enable_numba and NUMBA_AVAILABLE:
        shifts_x, shifts_y = _compute_pairwise_shifts_numba(
            i_x, i_y, j_x, j_y, max_shift_pixels, i_frames, j_frames, ton
        )
    else:
        shifts_x, shifts_y = _compute_pairwise_shifts_numba(
            i_x, i_y, j_x, j_y, max_shift_pixels, i_frames, j_frames, ton
        )  # Uses fallback

    phase1_time = time.time() - phase1_start
    phase2_start = time.time()

    # Create histogram bins
    n_bins = min(100, int(2 * max_shift_pixels))  # Adaptive bin count
    x_edges = np.linspace(
        -max_shift_pixels, max_shift_pixels, n_bins + 1
    ).astype(np.float32)
    y_edges = np.linspace(
        -max_shift_pixels, max_shift_pixels, n_bins + 1
    ).astype(np.float32)

    # Phase 2: Create histogram (Numba optimized)
    if enable_numba and NUMBA_AVAILABLE:
        hist = _histogram2d_numba(shifts_x, shifts_y, x_edges, y_edges)
    else:
        hist = _histogram2d_numba(
            shifts_x, shifts_y, x_edges, y_edges
        )  # Uses fallback

    phase2_time = time.time() - phase2_start
    phase3_start = time.time()

    # Phase 3: Find peak with sub-pixel precision
    peak_x_bin, peak_y_bin, peak_value = _find_histogram_peak_numba(hist)

    # Convert bin indices to shift values
    if n_bins > 1:
        bin_size_x = x_edges[1] - x_edges[0]
        bin_size_y = y_edges[1] - y_edges[0]
        shift_x = x_edges[0] + peak_x_bin * bin_size_x
        shift_y = y_edges[0] + peak_y_bin * bin_size_y
    else:
        shift_x, shift_y = 0.0, 0.0

    phase3_time = time.time() - phase3_start
    total_time = time.time() - start_time

    # Quality metrics
    quality_metrics = {
        "success": True,
        "peak_value": int(peak_value),
        "total_pairs": len(shifts_x),
        "n_frame_locs": len(
            locs_j
        ),  # locs_j is frame in standard call pattern
        "n_reference_locs": len(
            locs_i
        ),  # locs_i is reference in standard call pattern
        "numba_enabled": enable_numba and NUMBA_AVAILABLE,
        "sigma_x": np.std(shifts_x),
        "sigma_y": np.std(shifts_y),
        "timing": {
            "phase1_pairwise": phase1_time,
            "phase2_histogram": phase2_time,
            "phase3_peak": phase3_time,
            "total": total_time,
        },
    }

    # Create and save histogram plot if requested
    if plot_histogram and plot_dir is not None:
        plot_filepath = _save_rsso_histogram_plot(
            hist,
            x_edges,
            y_edges,
            shift_x,
            shift_y,
            max_shift_pixels,
            plot_dir,
            iteration,
            frame_number,
            quality_metrics,
        )
        quality_metrics["plot_filepath"] = plot_filepath

    return shift_x, shift_y, quality_metrics


def _save_rsso_histogram_plot(
    hist,
    x_edges,
    y_edges,
    shift_x,
    shift_y,
    max_shift,
    plot_dir,
    iteration,
    frame_number,
    quality_metrics,
):
    """
    Save 2D histogram plot showing RSSO shift distribution and peak.

    Args:
        hist : np.array
            2D histogram of shifts
        x_edges : np.array
            Histogram x bin edges
        y_edges : np.array
            Histogram y bin edges
        shift_x : float
            Estimated x shift
        shift_y : float
            Estimated y shift
        max_shift : float
            Maximum shift range
        plot_dir : str
            Directory to save plot
        iteration : int or None
            Iteration number for filename
        frame_number : int or None
            Frame number for filename
        quality_metrics : dict
            Quality metrics to display on plot

    Returns:
        filepath : str
            Path to saved plot
    """
    import matplotlib.pyplot as plt
    import random
    import string

    # Create rsso_plots subdirectory
    rsso_plot_dir = os.path.join(plot_dir, "rsso_plots")
    os.makedirs(rsso_plot_dir, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create coordinate grids - use bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers)

    # Apply circular mask for visualization
    hist_plot = hist.T.copy()
    if max_shift is not None:
        distances = np.sqrt(X_centers**2 + Y_centers**2)
        outside_circle = distances > max_shift
        hist_plot[outside_circle] = np.nan

    # Plot the 2D histogram
    im = ax.pcolormesh(
        X_centers, Y_centers, hist_plot, cmap="viridis", shading="nearest"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=20)

    # Add circular boundary
    if max_shift is not None:
        circle = plt.Circle(
            (0, 0),
            max_shift,
            fill=False,
            color="white",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(circle)

    # Mark the detected shift with a red cross
    ax.plot(
        shift_x,
        shift_y,
        "r+",
        markersize=15,
        markeredgewidth=2,
        label=f"Shift: ({shift_x:.3f}, {shift_y:.3f}) px",
    )

    # Set labels and title
    ax.set_xlabel("X Shift (pixels)")
    ax.set_ylabel("Y Shift (pixels)")

    # Build title with iteration and frame info
    title_parts = ["RSSO Shift Histogram"]
    if iteration is not None:
        title_parts.append(f"Iter {iteration}")
    if frame_number is not None:
        title_parts.append(f"Frame {frame_number}")
    ax.set_title(" - ".join(title_parts))

    # Set axis limits
    ax.set_xlim(-max_shift, max_shift)
    ax.set_ylim(-max_shift, max_shift)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Add text box with statistics
    total_points = quality_metrics.get("total_pairs", np.sum(hist))
    peak_count = quality_metrics.get("peak_value", np.max(hist))
    sigma_x = quality_metrics.get("sigma_x", 0)
    sigma_y = quality_metrics.get("sigma_y", 0)

    textstr = (
        f"Total pairs: {total_points:.0f}\n"
        f"Peak count: {peak_count:.0f}\n"
        f"σx: {sigma_x:.3f} px\n"
        f"σy: {sigma_y:.3f} px"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    # Generate filename with iteration and frame number
    filename_parts = ["rsso"]
    if iteration is not None:
        filename_parts.append(f"iter{iteration:02d}")
    if frame_number is not None:
        filename_parts.append(f"frame{frame_number:04d}")

    # Add random code for uniqueness
    rcode = "".join(random.choices(string.ascii_letters, k=6))
    filename_parts.append(rcode)

    filename = "_".join(filename_parts) + ".png"
    filepath = os.path.join(rsso_plot_dir, filename)

    # Save the plot
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    return filepath


def _validate_numba_implementation():
    """Validate that Numba implementation produces equivalent results to standard implementation"""
    from picasso_workflow.picasso_outpost import _calculate_pairwise_shift

    # Create test data with realistic RSSO structure
    np.random.seed(42)
    n_common = 800  # Points that appear in both datasets (with shift)
    n_frame_extra = 200  # Extra points only in frame
    n_ref_extra = 500  # Extra points only in reference

    true_shift_x, true_shift_y = 2.5, -1.8

    # Create common structure that appears in both datasets
    center_x, center_y = 50, 50
    noise_level = 0.3  # Small noise for realistic localization precision

    # Generate base pattern that will appear in both datasets
    np.random.seed(123)  # Different seed for base pattern
    base_x = np.random.normal(center_x, 4.0, n_common).astype(np.float32)
    base_y = np.random.normal(center_y, 4.0, n_common).astype(np.float32)

    # Frame dataset: base pattern + noise + extra points
    frame_x = base_x + np.random.normal(0, noise_level, n_common).astype(
        np.float32
    )
    frame_y = base_y + np.random.normal(0, noise_level, n_common).astype(
        np.float32
    )

    # Add some extra points only in frame
    if n_frame_extra > 0:
        extra_frame_x = np.random.uniform(45, 55, n_frame_extra).astype(
            np.float32
        )
        extra_frame_y = np.random.uniform(45, 55, n_frame_extra).astype(
            np.float32
        )
        frame_x = np.concatenate([frame_x, extra_frame_x])
        frame_y = np.concatenate([frame_y, extra_frame_y])

    # Reference dataset: shifted base pattern + noise + extra points
    ref_x = (base_x + true_shift_x) + np.random.normal(
        0, noise_level, n_common
    ).astype(np.float32)
    ref_y = (base_y + true_shift_y) + np.random.normal(
        0, noise_level, n_common
    ).astype(np.float32)

    # Add some extra points only in reference
    if n_ref_extra > 0:
        extra_ref_x = np.random.uniform(45, 60, n_ref_extra).astype(np.float32)
        extra_ref_y = np.random.uniform(45, 55, n_ref_extra).astype(np.float32)
        ref_x = np.concatenate([ref_x, extra_ref_x])
        ref_y = np.concatenate([ref_y, extra_ref_y])

    # Create recarray format for standard implementation
    frame_locs = np.rec.fromarrays([frame_x, frame_y], names=["x", "y"])
    ref_locs = np.rec.fromarrays([ref_x, ref_y], names=["x", "y"])

    max_shift_pixels = 10.0

    # Quick manual test: single point shift calculation
    manual_ref_x = np.array([50.0], dtype=np.float32)
    manual_ref_y = np.array([50.0], dtype=np.float32)
    manual_frame_x = np.array([52.5], dtype=np.float32)
    manual_frame_y = np.array([48.2], dtype=np.float32)

    manual_shifts_x, manual_shifts_y = _compute_pairwise_shifts_numba(
        manual_ref_x,
        manual_ref_y,
        manual_frame_x,
        manual_frame_y,
        max_shift_pixels,
    )
    print(
        f"    Manual test: expected shift (2.5, -1.8), got ({manual_shifts_x[0]:.1f}, {manual_shifts_y[0]:.1f})"
    )

    try:
        # Debug: Print test data statistics
        print(
            f"    Test data: {len(frame_locs)} frame locs, {len(ref_locs)} ref locs"
        )
        print(
            f"    Frame center: ({np.mean(frame_x):.1f}, {np.mean(frame_y):.1f})"
        )
        print(f"    Ref center: ({np.mean(ref_x):.1f}, {np.mean(ref_y):.1f})")
        print(
            f"    Expected center shift: ({np.mean(ref_x) - np.mean(frame_x):.3f}, {np.mean(ref_y) - np.mean(frame_y):.3f})"
        )

        # Test Numba implementation
        (
            numba_shift_x,
            numba_shift_y,
            numba_info,
        ) = _compute_rsso_shift_numba_optimized(
            ref_locs, frame_locs, max_shift_pixels
        )

        # Test standard implementation
        std_shift_x, std_shift_y, _, std_info = _calculate_pairwise_shift(
            ref_locs, frame_locs, max_shift_pixels, plot_histogram=False
        )

        # Debug: Print number of pairs processed
        if numba_info:
            print(
                f"    Numba processed {numba_info.get('total_pairs', 'unknown')} pairs"
            )
            print(
                f"    Numba histogram peak: {numba_info.get('peak_value', 'unknown')}"
            )
        print(f"    Max shift limit: {max_shift_pixels} pixels")

        # Test with identical small dataset to isolate the difference
        print(f"    Testing with identical subset...")
        small_frame = frame_locs[:10]  # First 10 points
        small_ref = ref_locs[:20]  # First 20 points

        (
            small_numba_x,
            small_numba_y,
            small_numba_info,
        ) = _compute_rsso_shift_numba_optimized(
            small_ref, small_frame, max_shift_pixels
        )
        (
            small_std_x,
            small_std_y,
            _,
            small_std_info,
        ) = _calculate_pairwise_shift(
            small_ref, small_frame, max_shift_pixels, plot_histogram=False
        )

        if small_numba_x is not None and small_std_x is not None:
            small_diff_x = abs(small_numba_x - small_std_x)
            small_diff_y = abs(small_numba_y - small_std_y)
            small_numba_pairs = (
                small_numba_info.get("total_pairs", 0)
                if small_numba_info
                else 0
            )
            print(
                f"    Small test: Numba ({small_numba_x:.3f}, {small_numba_y:.3f}), Standard ({small_std_x:.3f}, {small_std_y:.3f})"
            )
            print(
                f"    Small test diff: ({small_diff_x:.3f}, {small_diff_y:.3f}), pairs: {small_numba_pairs}"
            )
        else:
            print(f"    Small test failed - one implementation returned None")

        # Compare results
        if numba_shift_x is not None and std_shift_x is not None:
            diff_x = abs(numba_shift_x - std_shift_x)
            diff_y = abs(numba_shift_y - std_shift_y)

            tolerance = 1.0  # pixels (temporarily relaxed for debugging)

            # Debug output for troubleshooting
            print(
                f"    Debug: True shift was ({true_shift_x:.3f}, {true_shift_y:.3f})"
            )
            print(
                f"    Standard: ({std_shift_x:.3f}, {std_shift_y:.3f}), error=({abs(std_shift_x-true_shift_x):.3f}, {abs(std_shift_y-true_shift_y):.3f})"
            )
            print(
                f"    Numba:    ({numba_shift_x:.3f}, {numba_shift_y:.3f}), error=({abs(numba_shift_x-true_shift_x):.3f}, {abs(numba_shift_y-true_shift_y):.3f})"
            )

            if numba_info:
                print(
                    f"    Numba pairs processed: {numba_info.get('total_pairs', 'unknown')}"
                )

            if diff_x < tolerance and diff_y < tolerance:
                print(
                    f"    Numba validation PASSED: shifts agree within {tolerance} pixels"
                )
                if numba_info and "timing" in numba_info:
                    print(
                        f"    Numba computation time: {numba_info['timing']['total']:.4f}s"
                    )
                return True
            else:
                print(
                    f"    Numba validation FAILED: shifts differ by ({diff_x:.3f}, {diff_y:.3f}) pixels"
                )
                return False
        else:
            print(
                "    Numba validation FAILED: one or both implementations returned None"
            )
            return False

    except Exception as e:
        print(f"    Numba validation FAILED: {e}")
        return False


def _estimate_subsampling_uncertainty(
    frame_locs,
    reference_dataset,
    max_shift,
    subsampling_fraction,
    n_trials=3,
    enable_numba_optimization=True,
):
    """Estimate uncertainty added by subsampling via multiple subset trials

    Args:
        frame_locs : ndarray
            Localizations from single frame
        reference_dataset : ndarray
            Full reference dataset (already subsampled from self.locs)
        max_shift : float
            Maximum shift for RSSO computation
        subsampling_fraction : float
            Fraction to further subsample reference dataset
        n_trials : int
            Number of different subsets to test
        enable_numba_optimization : bool
            Whether to use Numba-optimized RSSO computation

    Returns:
        tuple : (mean_shift_x, mean_shift_y, uncertainty_x, uncertainty_y, confidence)
    """
    from picasso_workflow.picasso_outpost import _calculate_pairwise_shift

    if len(reference_dataset) == 0 or len(frame_locs) == 0:
        return None, None, np.inf, np.inf, 0.0

    shift_estimates = []
    n_subset = max(1000, int(len(reference_dataset) * subsampling_fraction))
    n_subset = min(
        n_subset, len(reference_dataset)
    )  # Don't exceed available data

    for trial in range(n_trials):
        # Different random subset for each trial
        np.random.seed(1000 + trial)
        if n_subset < len(reference_dataset):
            subset_indices = np.random.choice(
                len(reference_dataset), n_subset, replace=False
            )
            subset_dataset = reference_dataset[subset_indices]
        else:
            subset_dataset = reference_dataset

        # Calculate RSSO shift with this subset
        if enable_numba_optimization:
            # Use Numba-optimized RSSO computation
            (
                shift_x,
                shift_y,
                uncertainty_info,
            ) = _compute_rsso_shift_numba_optimized(
                subset_dataset, frame_locs, max_shift
            )
        else:
            # Use standard RSSO computation
            shift_x, shift_y, _, uncertainty_info = _calculate_pairwise_shift(
                subset_dataset,
                frame_locs,
                max_shift,
                plot_histogram=False,
            )

        if shift_x is not None and shift_y is not None:
            shift_estimates.append((shift_x, shift_y))

    if len(shift_estimates) >= 2:
        shifts_array = np.array(shift_estimates)

        # Calculate uncertainty as standard deviation across trials
        uncertainty_x = np.std(shifts_array[:, 0])
        uncertainty_y = np.std(shifts_array[:, 1])

        # Mean estimate
        mean_shift_x = np.mean(shifts_array[:, 0])
        mean_shift_y = np.mean(shifts_array[:, 1])

        # Confidence based on consistency and number of localizations
        uncertainty_magnitude = np.sqrt(uncertainty_x**2 + uncertainty_y**2)
        consistency_confidence = 1.0 / (
            1.0 + uncertainty_magnitude * 10
        )  # Penalize inconsistency
        size_confidence = min(1.0, len(frame_locs) / 100.0)
        confidence = consistency_confidence * size_confidence

        return (
            mean_shift_x,
            mean_shift_y,
            uncertainty_x,
            uncertainty_y,
            confidence,
        )
    elif len(shift_estimates) == 1:
        # Only one successful estimate
        shift_x, shift_y = shift_estimates[0]
        return (
            shift_x,
            shift_y,
            np.nan,
            np.nan,
            min(1.0, len(frame_locs) / 100.0),
        )
    else:
        return None, None, np.inf, np.inf, 0.0


def _compute_frame_to_reference_shift_optimized(frame_data):
    """Optimized RSSO shift computation with temporal filtering

    Args:
        frame_data : tuple
            (frame_indices, reference_dataset, target_frames, frame_locs, max_shift, min_locs_per_frame,
             enable_uncertainty_estimation, n_uncertainty_trials, subsampling_fraction,
             enable_numba_optimization, plot_histogram, plot_dir, iteration, ton)

    Returns:
        tuple : (frame_indices, shift_x, shift_y, uncertainty_x, uncertainty_y, confidence, quality, performance_info)
    """
    from picasso_workflow.picasso_outpost import _calculate_pairwise_shift

    (
        frame_indices,
        reference_dataset,
        target_frames,
        frame_locs,
        max_shift,
        min_locs_per_frame,
        enable_uncertainty_estimation,
        n_uncertainty_trials,
        subsampling_fraction,
        enable_numba_optimization,
        plot_histogram,
        plot_dir,
        iteration,
        ton,  # For temporal filtering
    ) = frame_data

    try:
        # Initialize uncertainty_info at the start to avoid "referenced before assignment" error
        uncertainty_info = {}

        # Skip frames with insufficient localizations
        if len(frame_locs) < min_locs_per_frame:
            logger.debug(
                f"Too few locs in frame group {target_frames}: {len(frame_locs)} < {min_locs_per_frame}"
            )
            return (frame_indices, None, None, None, None, 0.0, 0.0, None)

        # Create dataset by excluding all target frames
        from scipy.spatial import cKDTree

        # Use pre-built cKDTree and frame array from worker initialization if available
        global _WORKER_KDTREE, _WORKER_FRAMES
        if _WORKER_KDTREE is not None:
            # Worker has pre-built cKDTree - use it directly
            # Temporal filtering will be done at pair level using _WORKER_FRAMES
            dataset_locs = _WORKER_KDTREE
            len_dataset = _WORKER_KDTREE.n
        elif not isinstance(reference_dataset, cKDTree):
            # Pickle mode or sequential: filter reference_dataset by frame
            # Apply both exact frame exclusion AND temporal filtering
            dataset_mask = ~np.isin(reference_dataset["frame"], target_frames)

            # Add temporal filtering: exclude frames within ±2×ton of target frames
            if ton > 0:
                temporal_exclusion = 2 * ton
                for target_frame in target_frames:
                    # Exclude all frames in the range [target_frame - 2*ton, target_frame + 2*ton]
                    frame_diffs = np.abs(reference_dataset["frame"] - target_frame)
                    temporal_mask = frame_diffs > temporal_exclusion
                    dataset_mask &= temporal_mask

            dataset_locs = reference_dataset[dataset_mask]
            len_dataset = len(dataset_locs)
        else:
            # reference_dataset is already a cKDTree (shouldn't happen in normal flow)
            dataset_locs = reference_dataset
            len_dataset = reference_dataset.n

        if len_dataset == 0:
            logger.debug(
                f"No locs left in reference after masking out frame group {target_frames}"
            )
            return (frame_indices, None, None, None, None, 0.0, 0.0, None)

        # Choose computation method based on uncertainty estimation setting
        if enable_uncertainty_estimation and n_uncertainty_trials > 1:
            # Use uncertainty estimation with multiple subsets
            (
                shift_x,
                shift_y,
                uncertainty_x,
                uncertainty_y,
                confidence,
            ) = _estimate_subsampling_uncertainty(
                frame_locs,
                dataset_locs,
                max_shift,
                subsampling_fraction,
                n_uncertainty_trials,
                enable_numba_optimization,
            )
            quality = len(frame_locs) + len_dataset

        else:
            # Standard single computation (faster)
            import time

            start_time = time.time()

            frame_number = target_frames[0] if len(target_frames) > 0 else None

            # Extract frame numbers for temporal filtering
            frame_locs_frames = frame_locs["frame"] if "frame" in frame_locs.dtype.names else None
            ref_frames = _WORKER_FRAMES  # From worker global (or None if not using shared memory)

            if enable_numba_optimization:
                # Use Numba-optimized RSSO computation with temporal filtering
                # Determine frame number for plotting (use first frame in group)
                (
                    shift_x,
                    shift_y,
                    numba_info,
                ) = _compute_rsso_shift_numba_optimized(
                    dataset_locs,
                    frame_locs,
                    max_shift,
                    enable_numba=True,
                    plot_histogram=plot_histogram,
                    plot_dir=plot_dir,
                    iteration=iteration,
                    frame_number=frame_number,
                    ref_frames=ref_frames,
                    frame_locs_frames=frame_locs_frames,
                    ton=ton,
                )
                uncertainty_info = numba_info if numba_info is not None else {}
                computation_type = "Numba-optimized"
            else:
                # Use standard RSSO computation with temporal filtering
                shift_x, shift_y, _, std_info = _calculate_pairwise_shift(
                    dataset_locs,
                    frame_locs,
                    max_shift,
                    plot_histogram=plot_histogram,
                    remove_zeroshift=True,
                    plot_dir=plot_dir,
                    plot_fn_suffix=f"_{iteration}_{frame_number}_dslocs{len_dataset}_tgtlocs{len(frame_locs)}",
                    ref_frames=ref_frames,
                    frame_locs_frames=frame_locs_frames,
                    ton_exclusion=ton,
                )
                uncertainty_info = std_info if std_info is not None else {}
                computation_type = "Standard"

            computation_time = time.time() - start_time
            # logger.debug(f'Calculated shift in {computation_time} (reference: {len_dataset}, frame: {len(frame_locs)})')

            # Add timing info to uncertainty_info
            if uncertainty_info is None:
                uncertainty_info = {}
            uncertainty_info["computation_time"] = computation_time
            uncertainty_info["computation_type"] = computation_type
            uncertainty_info["n_dataset_locs"] = len_dataset
            uncertainty_info["n_frame_locs"] = len(frame_locs)

            if shift_x is not None and shift_y is not None:
                # Extract uncertainty from RSSO calculation based on computation method
                if computation_type == "Numba-optimized":
                    # Numba implementation doesn't provide uncertainty estimates
                    uncertainty_x = np.nan
                    uncertainty_y = np.nan
                elif computation_type == "Standard":
                    # Standard implementation provides shift_x_error and shift_y_error
                    uncertainty_x = (
                        uncertainty_info.get("shift_x_error", np.nan)
                        if uncertainty_info
                        else np.nan
                    )
                    uncertainty_y = (
                        uncertainty_info.get("shift_y_error", np.nan)
                        if uncertainty_info
                        else np.nan
                    )
                else:
                    uncertainty_x = np.nan
                    uncertainty_y = np.nan

                # Calculate confidence
                n_locs_frame = len(frame_locs)
                if not (np.isnan(uncertainty_x) or np.isnan(uncertainty_y)):
                    uncertainty_magnitude = np.sqrt(
                        uncertainty_x**2 + uncertainty_y**2
                    )
                    confidence = min(
                        1.0,
                        (n_locs_frame / 100.0) / (1.0 + uncertainty_magnitude),
                    )
                else:
                    confidence = min(1.0, n_locs_frame / 100.0)

                quality = len(frame_locs) + len_dataset
            else:
                logger.debug(f"shift x or y is None.")
                return (frame_indices, None, None, None, None, 0.0, 0.0, None)

        return (
            frame_indices,
            shift_x,
            shift_y,
            uncertainty_x,
            uncertainty_y,
            confidence,
            quality,
            uncertainty_info,  # Include performance metrics
        )

    except Exception as e:
        raise e
        print(f"Error processing frame group {frame_indices}: {e}")
        return (frame_indices, None, None, None, None, 0.0, 0.0, None)


# ==============================================================================
# Main computation function
# ==============================================================================


def compute_undrift_rsso(locs, pixelsize, info, parameters, results_folder):
    """Compute iterative RSSO-based drift correction

    This is the main computation function that performs the full iterative RSSO
    drift correction algorithm.

    Args:
        locs : structured array
            Localization data with 'x', 'y', 'frame' fields
        pixelsize : float
            Camera pixel size in nm
        info : dict
            Metadata dictionary
        parameters : dict
            Algorithm parameters (see undrift_rsso method docstring for details)
        results_folder : str
            Path to folder for saving results

    Returns:
        locs_undrifted : structured array
            Drift-corrected localizations
        drift : ndarray
            Drift trajectory (n_frames x 2)
        results : dict
            Analysis results dictionary
    """
    import gc
    import psutil
    from picasso import io
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # from concurrent.futures import ProcessPoolExecutor

    # Note: OpenMP configuration is now done at module import time
    # (see module-level code after imports) to ensure environment
    # variables are set before any numerical libraries initialize

    # Extract parameters with defaults
    max_frames = parameters.get("max_frames", np.inf)
    logger.debug(f"Cropping data to {max_frames} for debug reasons")
    locs = locs[locs["frame"] < max_frames]

    ton = parameters["ton"]
    toff = parameters["toff"]
    max_shift_nm = parameters["max_shift"]  # User provides in nanometers
    max_shift_pixels = (
        max_shift_nm / pixelsize
    )  # Convert to pixels for internal use
    min_locs_per_frame = parameters.get("min_locs_per_frame", 10)
    max_iterations = parameters.get("max_iterations", 5)
    convergence_threshold = parameters.get("convergence_threshold", 0.1)
    save_locs = parameters.get("save_locs", True)
    plot_drift = parameters.get("plot_drift", True)
    plot_rsso = parameters.get("plot_rsso", False)

    # Memory management parameters
    chunk_size = parameters.get("chunk_size", 100)
    memory_limit_gb = parameters.get("memory_limit_gb", 8.0)
    enable_multiprocessing = parameters.get("enable_multiprocessing", True)
    n_processes = (
        parameters.get("n_processes", min(mp.cpu_count(), 4))
        if enable_multiprocessing
        else 1
    )

    # Log multiprocessing configuration
    if enable_multiprocessing and n_processes > 1:
        logger.info(f"Multiprocessing enabled: {n_processes} processes")
        logger.info(
            "OpenMP threading disabled at module level to avoid fork() conflicts"
        )
    else:
        logger.info("Sequential processing mode")

    # Performance optimization parameters
    subsampling_fraction = parameters.get("subsampling_fraction", 0.1)
    enable_uncertainty_estimation = parameters.get(
        "enable_uncertainty_estimation", True
    )
    n_uncertainty_trials = parameters.get("n_uncertainty_trials", 3)
    adaptive_subsampling = parameters.get("adaptive_subsampling", False)
    target_uncertainty_nm = parameters.get("target_uncertainty_nm", 0.05)

    # Progressive subsampling parameters
    final_iteration_full_dataset = parameters.get(
        "final_iteration_full_dataset", True
    )
    progressive_subsampling = parameters.get("progressive_subsampling", False)
    enable_numba_optimization = parameters.get(
        "enable_numba_optimization", True
    )
    progressive_subsampling_schedule = parameters.get(
        "progressive_subsampling_schedule", [0.05, 0.1, 0.25, 0.5, 1.0]
    )

    # cKDTree sharing method for multiprocessing (non-Numba only)
    kdtree_sharing_method = parameters.get("kdtree_sharing_method", "worker_init")
    if kdtree_sharing_method not in ["worker_init", "shared_memory", "pickle"]:
        logger.warning(
            f"Invalid kdtree_sharing_method '{kdtree_sharing_method}', "
            f"using 'worker_init'"
        )
        kdtree_sharing_method = "worker_init"
    logger.info(f"cKDTree sharing method: {kdtree_sharing_method}")

    # Analysis parameters
    confidence_threshold = parameters.get("confidence_threshold", 0.8)
    outlier_detection_enabled = parameters.get(
        "outlier_detection_enabled", True
    )
    outlier_z_threshold = parameters.get("outlier_z_threshold", 3.5)
    min_signal_to_noise = parameters.get("min_signal_to_noise", 0.5)
    windowing_enabled = parameters.get("windowing_enabled", True)
    window_size_range = parameters.get("window_size_range", (3, 20))

    # Monitor initial memory usage
    process = psutil.Process()
    initial_memory_gb = process.memory_info().rss / (1024**3)

    logger.debug(
        f"Iterative RSSO undrift: max_iterations={max_iterations}, "
        f"convergence_threshold={convergence_threshold:.3f} nm, chunk_size={chunk_size}"
    )
    logger.debug(
        f"Using {n_processes} processes, memory limit: {memory_limit_gb:.1f} GB"
    )
    if progressive_subsampling:
        logger.debug(
            f"Progressive subsampling enabled: {progressive_subsampling_schedule}"
        )
    else:
        logger.debug(
            f"Fixed subsampling: {subsampling_fraction:.1%} of dataset"
        )

    logger.debug(
        f"Uncertainty estimation: {enable_uncertainty_estimation}, final iteration full dataset: {final_iteration_full_dataset}"
    )
    logger.debug(f"Initial memory usage: {initial_memory_gb:.2f} GB")

    # Initialize results dictionary
    results = {}

    # Get frame range and ensure we have data
    if len(locs) == 0:
        # Handle empty dataset
        drift = np.array([[0.0, 0.0]])
        results["success"] = True
        results["drift_magnitude_x"] = 0.0
        results["drift_magnitude_y"] = 0.0
        results["total_drift"] = 0.0
        results["mean_drift_quality"] = 0.0
        return locs, drift, results

    frames = np.arange(locs["frame"].min(), locs["frame"].max() + 1)
    n_frames = len(frames)

    # Initialize arrays for iterative approach
    drift_x = np.zeros(n_frames)  # Total drift per frame
    drift_y = np.zeros(n_frames)
    uncertainty_x = np.zeros(n_frames)  # Uncertainty estimates
    uncertainty_y = np.zeros(n_frames)
    confidence = np.zeros(n_frames)  # Confidence measures
    drift_quality = np.zeros(n_frames)  # Quality metrics

    # Store original localization data to preserve for each iteration
    # We'll accumulate corrections without modifying the original dataset
    original_locs = locs.copy()
    cumulative_corrections_x = np.zeros(len(locs), dtype=np.float32)
    cumulative_corrections_y = np.zeros(len(locs), dtype=np.float32)

    # PRE-COMPUTE frame index mapping (optimization to avoid recomputing every iteration)
    # This maps each localization to its frame index for fast lookup
    frame_index_map = (original_locs["frame"] - frames[0]).astype(np.int32)
    logger.debug(f"Pre-computed frame index mapping for {len(frame_index_map):,} localizations")

    # Estimate memory requirements and warn if needed
    bytes_per_loc = locs.itemsize * len(locs.dtype)
    estimated_memory_gb = (len(locs) * bytes_per_loc * 3) / (
        1024**3
    )  # Factor for processing
    logger.debug(
        f"Dataset size: {len(locs):,} localizations, {n_frames:,} frames"
    )
    logger.debug(f"Estimated memory requirement: {estimated_memory_gb:.2f} GB")

    if estimated_memory_gb > memory_limit_gb:
        logger.debug(
            f"WARNING: Estimated memory ({estimated_memory_gb:.2f} GB) exceeds limit ({memory_limit_gb:.1f} GB)"
        )
        logger.debug(
            "Consider reducing chunk_size or increasing memory_limit_gb"
        )

    # Use in-place updates instead of copying (major memory saving)
    # No more: current_locs = self.locs.copy()

    logger.debug(
        f"Starting iterative RSSO undrift: {n_frames} frames, ton={ton}, toff={toff}"
    )
    logger.debug(
        f"Max shift: {max_shift_nm:.1f} nm ({max_shift_pixels:.2f} pixels)"
    )

    # Validate Numba implementation if enabled
    if enable_numba_optimization:
        logger.debug("  Validating Numba optimization...")
        enable_numba_optimization = _validate_numba_implementation()

    # Save original localizations if requested
    if save_locs:
        fp_locs = os.path.join(results_folder, "locs_original_input.hdf5")
        io.save_locs(fp_locs, locs, info)

    # Start iterative refinement loop
    iteration_history = []
    convergence_rms = float("inf")

    for iteration in range(max_iterations):
        # Initialize comprehensive timing dictionary for this iteration
        iteration_timings = {
            # Setup phase
            "reference_creation": 0.0,
            "kdtree_creation": 0.0,
            "kdtree_serialization": 0.0,

            # Multiprocessing setup
            "pool_creation": 0.0,
            "worker_initialization": 0.0,

            # Processing phase
            "frame_grouping": 0.0,
            "chunk_data_preparation": 0.0,
            "pool_map_total": 0.0,
            "result_collection": 0.0,
            "pool_teardown": 0.0,
            "frame_processing": 0.0,

            # Worker computation (extracted from performance_info)
            "worker_computation": 0.0,
            "multiprocessing_overhead": 0.0,

            # Post-processing phase
            "array_copy": 0.0,
            "frame_pregrouping": 0.0,
            "windowing_outliers": 0.0,
            "array_operations": 0.0,
            "frame_corrections": 0.0,

            # Finalization phase
            "convergence_check": 0.0,
            "history_storage": 0.0,

            # Totals and metadata
            "total": 0.0,
            "chunk_times": [],
            "worker_times": [],
        }

        # Start iteration timer
        iteration_start_time_perf = time.perf_counter()

        logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
        logger.debug(
            f"    Numba optimization: {'enabled' if enable_numba_optimization else 'disabled'}"
        )
        iter_dir = os.path.join(results_folder, f"iteration_{iteration + 1}")
        os.makedirs(iter_dir, exist_ok=True)

        # Monitor memory usage during iteration
        current_memory_gb = process.memory_info().rss / (1024**3)
        logger.debug(
            f"    Memory usage at iteration start: {current_memory_gb:.2f} GB"
        )

        if current_memory_gb > memory_limit_gb:
            logger.debug(
                f"    WARNING: Memory usage ({current_memory_gb:.2f} GB) exceeds limit!"
            )

        # Determine subsampling fraction for this iteration
        if final_iteration_full_dataset and iteration == max_iterations - 1:
            # Final iteration: always use full dataset for maximum accuracy
            current_subsampling_fraction = 1.0
            logger.debug(f"    Final iteration: using full dataset (100%)")
        elif progressive_subsampling:
            # Progressive subsampling: use schedule
            if iteration < len(progressive_subsampling_schedule):
                current_subsampling_fraction = (
                    progressive_subsampling_schedule[iteration]
                )
            else:
                # If we exceed schedule length, use last value
                current_subsampling_fraction = (
                    progressive_subsampling_schedule[-1]
                )
            logger.debug(
                f"    Progressive subsampling: {current_subsampling_fraction:.1%} of dataset"
            )
        else:
            # Fixed subsampling fraction
            current_subsampling_fraction = subsampling_fraction
            logger.debug(
                f"    Fixed subsampling: {current_subsampling_fraction:.1%} of dataset"
            )

        # Create current dataset with accumulated corrections for this iteration
        with _Timer("array_copy") as copy_timer:
            current_locs = original_locs.copy()
            current_locs["x"] = original_locs["x"] + cumulative_corrections_x
            current_locs["y"] = original_locs["y"] + cumulative_corrections_y
        iteration_timings["array_copy"] = copy_timer.elapsed
        logger.debug(f"    Array copy + corrections: {_format_time(copy_timer.elapsed)}")

        # OPTIMIZATION: Pre-group localizations by frame for fast lookup
        # This eliminates 15+ minutes of boolean masking operations per iteration
        with _Timer("frame_pregrouping") as pregroup_timer:
            # Sort by frame to enable slice-based access
            sort_indices = np.argsort(current_locs["frame"])
            current_locs = current_locs[sort_indices]

            # Compute frame boundaries using unique frames
            unique_frames, frame_start_indices, frame_counts = np.unique(
                current_locs["frame"], return_index=True, return_counts=True
            )

            # Build fast lookup dictionary: frame_num -> (start_idx, end_idx, count)
            frame_boundaries = {}
            for i, frame_num in enumerate(unique_frames):
                start_idx = frame_start_indices[i]
                count = frame_counts[i]
                end_idx = start_idx + count
                frame_boundaries[frame_num] = (start_idx, end_idx, count)

            logger.debug(
                f"    Pre-grouped {len(current_locs):,} locs into {len(unique_frames)} frames: {_format_time(pregroup_timer.elapsed)}"
            )
        iteration_timings["frame_pregrouping"] = pregroup_timer.elapsed

        # OPTIMIZATION: Create reference dataset for this iteration
        with _Timer("reference_creation") as ref_timer:
            if current_subsampling_fraction < 1.0:
                # subsample by cropping the center
                x_min_full, x_max_full = (
                    current_locs["x"].min(),
                    current_locs["x"].max(),
                )
                dx_full = x_max_full - x_min_full
                y_min_full, y_max_full = (
                    current_locs["y"].min(),
                    current_locs["y"].max(),
                )
                dy_full = y_max_full - y_min_full

                # logger.debug(f"full FOV: ({x_min_full} - {x_max_full}, {y_min_full} - {y_max_full})")

                dx_reduced = dx_full * np.sqrt(current_subsampling_fraction)
                dy_reduced = dy_full * np.sqrt(current_subsampling_fraction)

                # logger.debug(f"reducing side lengths from ({dx_full}, {dy_full}) to ({dx_reduced}, {dy_reduced})")

                x_min = x_min_full + (dx_full - dx_reduced) / 2
                x_max = x_max_full - (dx_full - dx_reduced) / 2
                y_min = y_min_full + (dy_full - dy_reduced) / 2
                y_max = y_max_full - (dy_full - dy_reduced) / 2

                # logger.debug(f"reduced FOV: ({x_min} - {x_max}, {y_min} - {y_max})")
                # logger.debug(f'#locs > xmin: {(current_locs["x"] > x_min).sum()}')
                # logger.debug(f'#locs < xmax: {(current_locs["x"] < x_max).sum()}')
                # logger.debug(f'#locs > y_min: {(current_locs["y"] > y_min).sum()}')
                # logger.debug(f'#locs < y_min: {(current_locs["y"] < y_min).sum()}')
                # logger.debug(f'#locs in xrange: {((current_locs["x"] > x_min) & (current_locs["x"] < x_max)).sum()}')
                # logger.debug(f'#locs in yrange: {((current_locs["y"] > y_min) & (current_locs["y"] < y_max)).sum()}')
                # logger.debug(f'#locs in reduced FOV: {((current_locs["x"] > x_min) & (current_locs["x"] < x_max) & (current_locs["y"] > y_min) & (current_locs["y"] < y_max)).sum()}')

                # indices = (
                #     (current_locs["x"] > x_min)
                #     & (current_locs["x"] < x_max)
                #     & (current_locs["y"] > y_min)
                #     & (current_locs["y"] < y_min))
                # indices = (
                #     (current_locs["x"] > x_min)
                #     & (current_locs["x"] < x_max)
                #     & (current_locs["y"] > y_min)
                #     & (current_locs["x"] < y_max))
                indices = (
                    (current_locs["x"] > x_min)
                    & (current_locs["x"] < x_max)
                    & (current_locs["y"] > y_min)
                    & (current_locs["y"] < y_max)
                )
                # only crop if there are enough locs
                # logger.debug(f"found {indices.sum()} locs in range. Minimum is {min_locs_per_frame}")
                if indices.sum() > min_locs_per_frame:
                    current_locs = current_locs[indices]

                reference_dataset = current_locs

                logger.debug(
                    f"    Created reference dataset: {len(reference_dataset):,} locs ({current_subsampling_fraction:.1%} of full dataset)"
                )
            else:
                reference_dataset = current_locs
                logger.debug(
                    f"    Using full dataset as reference: {len(reference_dataset):,} locs"
                )
        iteration_timings["reference_creation"] = ref_timer.elapsed

        # Prepare reference data for multiprocessing
        # Extract coordinates for worker initialization (if using cKDTree approach)
        if not enable_numba_optimization:
            with _Timer("kdtree_creation") as kdtree_timer:
                reference_coords = np.column_stack(
                    [reference_dataset.x, reference_dataset.y]
                )
                # For backwards compatibility, also build cKDTree here
                # (will be overridden by worker-level trees when using multiprocessing)
                from scipy.spatial import cKDTree

                reference_dataset_kdtree = cKDTree(reference_coords)
            # Log after context manager exits
            logger.debug(
                f"    cKDTree creation: {_format_time(kdtree_timer.elapsed)}"
            )
            iteration_timings["kdtree_creation"] = kdtree_timer.elapsed
        else:
            reference_coords = None
            reference_dataset_kdtree = None

        # Process frames in chunks using same reference dataset
        logger.debug(
            f"    Processing {n_frames} frames in chunks of {chunk_size}..."
        )

        # Initialize results for this iteration
        frame_shifts_x = np.zeros(n_frames)
        frame_shifts_y = np.zeros(n_frames)
        new_uncertainty_x = np.zeros(n_frames)
        new_uncertainty_y = np.zeros(n_frames)
        new_sigma_x = np.zeros(n_frames)
        new_sigma_y = np.zeros(n_frames)
        new_confidence = np.zeros(n_frames)
        new_quality = np.zeros(n_frames)
        valid_measurements = 0

        # Performance monitoring
        iteration_start_time = time.time()
        numba_computation_times = []
        standard_computation_times = []
        n_numba_computations = 0
        n_standard_computations = 0

        # OPTIMIZATION: Create pool and shared memory ONCE for all chunks
        # This saves ~30 seconds per iteration by avoiding repeated setup/teardown
        pool = None
        shm = None
        frame_shm = None  # For temporal filtering
        ctx = None
        if enable_multiprocessing and n_processes > 1:
            try:
                ctx = _setup_multiprocessing_context()

                # Setup pool and shared memory based on method
                if reference_coords is not None:
                    if kdtree_sharing_method == "shared_memory":
                        logger.debug(
                            f"    Creating shared memory cKDTree with {reference_coords.shape[0]:,} points (ONCE for all chunks)"
                        )
                        with _Timer("kdtree_serial") as serial_timer:
                            shm, kdtree_size = _create_shared_memory_kdtree(
                                reference_coords
                            )
                        iteration_timings["kdtree_serialization"] += serial_timer.elapsed

                        # Also create shared memory for frame array (temporal filtering)
                        reference_frames = reference_dataset["frame"]
                        frame_shm, frame_len, frame_dtype = _create_shared_memory_frame_array(
                            reference_frames
                        )

                        with _Timer("pool_init") as pool_timer:
                            pool = ctx.Pool(
                                processes=n_processes,
                                initializer=_init_worker_from_shared_memory,
                                initargs=(shm.name, kdtree_size, frame_shm.name, frame_len, frame_dtype),
                            )
                        iteration_timings["pool_creation"] += pool_timer.elapsed

                    elif kdtree_sharing_method == "worker_init":
                        logger.debug(
                            f"    Initializing {n_processes} workers to build cKDTree "
                            f"({reference_coords.shape[0]:,} points each, ONCE for all chunks)"
                        )
                        with _Timer("pool_init") as pool_timer:
                            pool = ctx.Pool(
                                processes=n_processes,
                                initializer=_worker_init_kdtree,
                                initargs=(reference_coords,),
                            )
                        iteration_timings["pool_creation"] += pool_timer.elapsed

                    else:  # "pickle"
                        logger.debug(
                            f"    Creating pool for pickle mode (ONCE for all chunks)"
                        )
                        with _Timer("pool_init") as pool_timer:
                            pool = ctx.Pool(processes=n_processes)
                        iteration_timings["pool_creation"] += pool_timer.elapsed
                else:
                    # Numba mode - no KDTree
                    logger.debug(
                        f"    Creating pool for Numba mode (ONCE for all chunks)"
                    )
                    with _Timer("pool_init") as pool_timer:
                        pool = ctx.Pool(processes=n_processes)
                    iteration_timings["pool_creation"] += pool_timer.elapsed
            except Exception as e:
                logger.debug(f"    ⚠ Failed to create pool: {e}, falling back to sequential")
                enable_multiprocessing = False

        # Process frames in chunks (reusing the same pool)
        for chunk_start in range(0, n_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            chunk_frames = range(chunk_start, chunk_end)

            logger.debug(
                f"      Processing chunk {chunk_start//chunk_size + 1}/{(n_frames-1)//chunk_size + 1}: frames {chunk_start}-{chunk_end-1}"
            )

            # Evaluate frame sizes and create frame groups to ensure min_locs_per_frame
            # OPTIMIZATION: Use pre-computed frame_boundaries for O(1) lookup instead of O(n) boolean masking
            with _Timer("frame_grouping") as grouping_timer:
                frame_groups = (
                    []
                )  # List of (frame_indices, combined_frame_numbers)
                current_group = []
                current_locs_count = 0

                for frame_idx in chunk_frames:
                    frame_number = frames[frame_idx]
                    # OPTIMIZED: O(1) dict lookup instead of O(n) boolean mask
                    if frame_number in frame_boundaries:
                        _, _, frame_locs_count = frame_boundaries[frame_number]
                    else:
                        frame_locs_count = 0

                    current_group.append(frame_idx)
                    current_locs_count += frame_locs_count

                    # If we have enough locs or this is the last frame in chunk, finalize group
                    if (
                        current_locs_count >= min_locs_per_frame
                        or frame_idx == chunk_frames[-1]
                    ):

                        # Get all frame numbers in this group
                        group_frame_numbers = [
                            frames[idx] for idx in current_group
                        ]
                        frame_groups.append(
                            (current_group.copy(), group_frame_numbers)
                        )

                        # logger.debug(
                        #     f"        Created frame group: indices {current_group} "
                        #     f"(frames {group_frame_numbers}), {current_locs_count} locs"
                        #     )

                        # Start new group
                        current_group = []
                        current_locs_count = 0
            iteration_timings["frame_grouping"] += grouping_timer.elapsed

            # Prepare chunk data using frame groups
            # OPTIMIZATION: Use slice indexing instead of boolean masking
            with _Timer("data_prep") as prep_timer:
                chunk_frame_data = []
                for group_indices, group_frame_numbers in frame_groups:
                    # Extract frame localizations using pre-computed boundaries
                    # OPTIMIZED: O(1) slice instead of O(n) boolean mask
                    if group_frame_numbers:
                        # Get slice boundaries for all frames in the group
                        first_frame = group_frame_numbers[0]
                        last_frame = group_frame_numbers[-1]

                        if first_frame in frame_boundaries and last_frame in frame_boundaries:
                            start_idx = frame_boundaries[first_frame][0]
                            end_idx = frame_boundaries[last_frame][1]
                            frame_locs = current_locs[start_idx:end_idx]
                        else:
                            # Fallback if frame not in boundaries (shouldn't happen)
                            frame_mask = np.isin(
                                current_locs["frame"], group_frame_numbers
                            )
                            frame_locs = current_locs[frame_mask]
                    else:
                        frame_locs = np.array([], dtype=current_locs.dtype)

                    # Decide which reference to pass based on multiprocessing mode
                    # Different strategies require different data to be passed to workers
                    if (
                        enable_multiprocessing
                        and n_processes > 1
                        and reference_coords is not None
                        and kdtree_sharing_method != "pickle"
                    ):
                        # worker_init or shared_memory: workers have pre-built cKDTree
                        ref_data_for_worker = None
                    elif reference_dataset_kdtree is not None:
                        # Sequential processing or pickle mode with cKDTree
                        ref_data_for_worker = reference_dataset_kdtree
                    else:
                        # Numba optimization or fallback to raw data
                        ref_data_for_worker = reference_dataset

                    frame_data = (
                        group_indices,  # List of frame indices this result applies to
                        ref_data_for_worker,  # Reference dataset (None if using worker cKDTree)
                        group_frame_numbers,  # List of frame numbers to process together
                        frame_locs,
                        max_shift_pixels,
                        min_locs_per_frame,
                        enable_uncertainty_estimation,
                        n_uncertainty_trials,
                        current_subsampling_fraction,  # For uncertainty estimation subsampling
                        enable_numba_optimization,  # Use Numba JIT acceleration
                        plot_rsso,  # Whether to plot RSSO histograms
                        iter_dir,  # Directory for saving plots
                        iteration + 1,  # Current iteration number (1-indexed)
                        ton,  # For temporal filtering (exclude frames within ±2×ton)
                    )
                    chunk_frame_data.append(frame_data)
            iteration_timings["chunk_data_preparation"] += prep_timer.elapsed

            # Process chunk using pre-created pool (OPTIMIZED: pool reused across all chunks)
            with _Timer("chunk_processing") as chunk_timer:
                if pool is not None:
                    # Use pre-created pool for parallel processing
                    with _Timer("pool_map") as map_timer:
                        chunk_results = pool.map(
                            _compute_frame_to_reference_shift_optimized,
                            chunk_frame_data,
                        )
                    iteration_timings["pool_map_total"] += map_timer.elapsed
                else:
                    # Sequential processing
                    chunk_results = [
                        _compute_frame_to_reference_shift_optimized(frame_data)
                        for frame_data in chunk_frame_data
                    ]

            # Record chunk timing
            iteration_timings["chunk_times"].append(chunk_timer.elapsed)
            iteration_timings["frame_processing"] += chunk_timer.elapsed

            # Process chunk results immediately to avoid accumulating large arrays
            with _Timer("result_collection") as collection_timer:
                for result in chunk_results:
                    (
                        frame_indices,  # Now a list of frame indices
                        shift_x,
                        shift_y,
                        uncertainty_x_val,
                        uncertainty_y_val,
                        confidence_val,
                        quality_val,
                        performance_info,
                    ) = result
                    # logger.debug(f"chunk results: {result}")

                    # Collect performance statistics
                    if performance_info and "computation_time" in performance_info:
                        comp_time = performance_info["computation_time"]
                        comp_type = performance_info.get(
                            "computation_type", "Unknown"
                        )

                        if comp_type == "Numba-optimized":
                            numba_computation_times.append(comp_time)
                            n_numba_computations += 1
                        elif comp_type == "Standard":
                            standard_computation_times.append(comp_time)
                            n_standard_computations += 1

                        # Collect worker computation times for overhead calculation
                        iteration_timings["worker_times"].append(comp_time)

                    if shift_x is not None and shift_y is not None:
                        # Apply the same shift to all frames in the group
                        for frame_idx in frame_indices:
                            frame_shifts_x[frame_idx] = shift_x
                            frame_shifts_y[frame_idx] = shift_y
                            new_uncertainty_x[frame_idx] = uncertainty_x_val
                            new_uncertainty_y[frame_idx] = uncertainty_y_val
                            new_sigma_x[frame_idx] = performance_info["sigma_x"]
                            new_sigma_y[frame_idx] = performance_info["sigma_y"]
                            new_confidence[frame_idx] = confidence_val
                            new_quality[frame_idx] = quality_val
                            valid_measurements += 1
            iteration_timings["result_collection"] += collection_timer.elapsed

            # Force garbage collection after each chunk
            gc.collect()

        # OPTIMIZATION: Cleanup pool and shared memory after ALL chunks are processed
        if pool is not None:
            with _Timer("pool_teardown") as teardown_timer:
                pool.close()
                pool.join()
            iteration_timings["pool_teardown"] += teardown_timer.elapsed
            logger.debug("    Pool closed and cleaned up")

        if shm is not None:
            shm.close()
            shm.unlink()
            logger.debug("    KDTree shared memory cleaned up")

        if frame_shm is not None:
            frame_shm.close()
            frame_shm.unlink()
            logger.debug("    Frame array shared memory cleaned up")

        # Convert pixel shifts to nm and finalize arrays
        with _Timer("array_operations") as array_ops_timer:
            frame_shifts_x *= pixelsize  # Convert to nm
            frame_shifts_y *= pixelsize
            new_uncertainty_x *= pixelsize
            new_uncertainty_y *= pixelsize
            new_sigma_x *= pixelsize
            new_sigma_y *= pixelsize
        iteration_timings["array_operations"] = array_ops_timer.elapsed

        logger.debug(
            f"    Valid measurements: {valid_measurements}/{n_frames}"
        )

        # Report subsampling performance
        if current_subsampling_fraction < 1.0:
            speedup_estimate = 1.0 / current_subsampling_fraction
            logger.debug(
                f"    Estimated speedup from subsampling: {speedup_estimate:.1f}x"
            )

        # Report uncertainty statistics if enabled
        if enable_uncertainty_estimation:
            valid_uncertainties = new_uncertainty_x[new_uncertainty_x > 0]
            if len(valid_uncertainties) > 0:
                mean_uncertainty = np.mean(valid_uncertainties)
                logger.debug(
                    f"    Mean subsampling uncertainty: {mean_uncertainty:.3f} nm"
                )

                # Check if we should adjust subsampling fraction for next iteration
                if (
                    adaptive_subsampling
                    and mean_uncertainty > target_uncertainty_nm * 2
                ):
                    suggested_fraction = min(
                        1.0, current_subsampling_fraction * 1.5
                    )
                    logger.debug(
                        f"    High uncertainty detected - consider increasing subsampling_fraction to {suggested_fraction:.2f}"
                    )
                elif (
                    adaptive_subsampling
                    and mean_uncertainty < target_uncertainty_nm * 0.5
                ):
                    suggested_fraction = max(
                        0.05, current_subsampling_fraction * 0.8
                    )
                    logger.debug(
                        f"    Low uncertainty detected - could reduce subsampling_fraction to {suggested_fraction:.2f} for speed"
                    )

        # Handle outliers and windowing for low-confidence measurements
        with _Timer("windowing_outliers") as window_timer:
            if windowing_enabled:
                # Apply windowing to low-confidence frames
                low_confidence_mask = new_confidence < confidence_threshold
                n_low_conf = np.sum(low_confidence_mask)
                if n_low_conf > 0:
                    logger.debug(
                        f"    Applying windowing to {n_low_conf} low-confidence frames"
                    )
                    # For low-confidence frames, use windowed averaging (simplified approach)
                    min_window, max_window = window_size_range
                    for frame_idx in np.where(low_confidence_mask)[0]:
                        if frame_idx > 0:  # Use previous frame's shift as fallback
                            frame_shifts_x[frame_idx] = (
                                frame_shifts_x[frame_idx - 1] * 0.5
                            )
                            frame_shifts_y[frame_idx] = (
                                frame_shifts_y[frame_idx - 1] * 0.5
                            )
                            new_confidence[frame_idx] = 0.5

            # Outlier detection using z-score
            if outlier_detection_enabled and valid_measurements > 5:
                shifts_magnitude = np.sqrt(frame_shifts_x**2 + frame_shifts_y**2)
                valid_shifts = shifts_magnitude[shifts_magnitude > 0]
                if len(valid_shifts) > 0:
                    z_scores = np.abs(
                        (shifts_magnitude - np.mean(valid_shifts))
                        / np.std(valid_shifts)
                    )
                    outliers = z_scores > outlier_z_threshold
                    n_outliers = np.sum(outliers)
                    if n_outliers > 0:
                        logger.debug(
                            f"    Detected and filtered {n_outliers} outliers"
                        )
                        # Set outlier shifts to zero
                        frame_shifts_x[outliers] = 0
                        frame_shifts_y[outliers] = 0
                        new_confidence[outliers] = 0
        iteration_timings["windowing_outliers"] += window_timer.elapsed
        logger.debug(f"    Windowing + outlier detection: {_format_time(window_timer.elapsed)}")

        # mean frame shift should be 0 to keep the image in place
        with _Timer("mean_centering") as center_timer:
            frame_shifts_x -= np.mean(frame_shifts_x)
            frame_shifts_y -= np.mean(frame_shifts_y)
        iteration_timings["array_operations"] += center_timer.elapsed
        # Update cumulative drift arrays (already in nm from conversion above)
        drift_x += frame_shifts_x
        drift_y += frame_shifts_y
        uncertainty_x = new_uncertainty_x.copy()
        uncertainty_y = new_uncertainty_y.copy()
        sigma_x = new_sigma_x.copy()
        sigma_y = new_sigma_y.copy()
        confidence = new_confidence.copy()
        drift_quality = new_quality.copy()

        # Accumulate drift corrections for next iteration
        # Convert frame-based shifts to per-localization corrections
        with _Timer("frame_corrections") as corrections_timer:
            # Use optimized Numba function (or fallback NumPy if Numba unavailable)
            frame_corrections_x, frame_corrections_y = _apply_frame_corrections_numba(
                frame_shifts_x, frame_shifts_y, frame_index_map, pixelsize
            )

            # Accumulate corrections using in-place operations
            np.subtract(
                cumulative_corrections_x, frame_corrections_x, out=cumulative_corrections_x
            )
            np.subtract(
                cumulative_corrections_y, frame_corrections_y, out=cumulative_corrections_y
            )
        iteration_timings["frame_corrections"] = corrections_timer.elapsed
        logger.debug(f"    Frame corrections application: {_format_time(corrections_timer.elapsed)}")

        # Check for convergence
        with _Timer("convergence_check") as convergence_timer:
            if iteration > 0:
                # Calculate RMS change from previous iteration
                prev_drift_x, prev_drift_y = (
                    iteration_history[-1]["drift_x"],
                    iteration_history[-1]["drift_y"],
                )
                rms_change_x = np.sqrt(np.mean((drift_x - prev_drift_x) ** 2))
                rms_change_y = np.sqrt(np.mean((drift_y - prev_drift_y) ** 2))
                convergence_rms = np.sqrt(rms_change_x**2 + rms_change_y**2)

                logger.debug(f"    RMS change: {convergence_rms:.3f} nm")

                if convergence_rms < convergence_threshold:
                    logger.debug(
                        f"    ✓ Converged after {iteration + 1} iterations (RMS change < {convergence_threshold:.3f} nm)"
                    )
                    break
            else:
                rms_change_x = np.sqrt(np.mean((drift_x) ** 2))
                rms_change_y = np.sqrt(np.mean((drift_y) ** 2))
                convergence_rms = np.sqrt(rms_change_x**2 + rms_change_y**2)

                logger.debug(f"    RMS drift: {convergence_rms:.3f} nm")

                if convergence_rms < convergence_threshold:
                    logger.debug(
                        f"    ✓ Converged after {iteration + 1} iterations (RMS change < {convergence_threshold:.3f} nm)"
                    )
                    break
        iteration_timings["convergence_check"] = convergence_timer.elapsed

        # Store iteration history
        with _Timer("history_storage") as history_timer:
            iteration_history.append(
                {
                    "iteration": iteration + 1,
                    "drift_x": drift_x.copy(),
                    "drift_y": drift_y.copy(),
                    "uncertainty_x": new_uncertainty_x.copy(),
                    "uncertainty_y": new_uncertainty_y.copy(),
                    "sigma_x": new_sigma_x.copy(),
                    "sigma_y": new_sigma_y.copy(),
                    "confidence": new_confidence.copy(),
                    "convergence_rms": convergence_rms,
                    "valid_measurements": valid_measurements,
                }
            )
        iteration_timings["history_storage"] = history_timer.elapsed

        # Performance reporting for this iteration
        iteration_end_time_perf = time.perf_counter()
        iteration_timings["total"] = iteration_end_time_perf - iteration_start_time_perf

        # Calculate worker computation time and multiprocessing overhead
        if iteration_timings["worker_times"]:
            iteration_timings["worker_computation"] = sum(iteration_timings["worker_times"])
            # Overhead = total pool_map time - actual worker computation time
            # This includes serialization, IPC, and scheduling overhead
            if iteration_timings["pool_map_total"] > 0:
                iteration_timings["multiprocessing_overhead"] = (
                    iteration_timings["pool_map_total"]
                    - iteration_timings["worker_computation"] / n_processes
                )

        # Legacy aggregation metric (deprecated - use fine-grained metrics instead)
        iteration_timings["aggregation"] = (
            iteration_timings["total"]
            - iteration_timings["kdtree_creation"]
            - iteration_timings["frame_processing"]
        )

        # Log performance summary
        _log_performance_summary(
            iteration + 1,
            n_frames,
            iteration_timings,
            kdtree_sharing_method if not enable_numba_optimization else "numba",
            n_processes if enable_multiprocessing else 1,
        )

        logger.debug(f"    Iteration {iteration + 1} completed")

        # Report Numba vs Standard performance
        if enable_numba_optimization and numba_computation_times:
            avg_numba_time = np.mean(numba_computation_times)
            total_numba_time = np.sum(numba_computation_times)
            logger.debug(
                f"    Numba computations: {n_numba_computations}, avg {avg_numba_time:.4f}s, total {total_numba_time:.1f}s"
            )

            if standard_computation_times:
                avg_standard_time = np.mean(standard_computation_times)
                total_standard_time = np.sum(standard_computation_times)
                speedup = (
                    avg_standard_time / avg_numba_time
                    if avg_numba_time > 0
                    else 0
                )
                logger.debug(
                    f"    Standard computations: {n_standard_computations}, avg {avg_standard_time:.4f}s, total {total_standard_time:.1f}s"
                )
                logger.debug(f"    Numba speedup: {speedup:.1f}x")
        elif not enable_numba_optimization and standard_computation_times:
            avg_standard_time = np.mean(standard_computation_times)
            total_standard_time = np.sum(standard_computation_times)
            logger.debug(
                f"    Standard computations: {n_standard_computations}, avg {avg_standard_time:.4f}s, total {total_standard_time:.1f}s"
            )

    # Finalize results
    n_iterations = len(iteration_history)
    final_convergence_rms = (
        convergence_rms if n_iterations > 1 else float("inf")
    )

    # Report final memory usage
    final_memory_gb = process.memory_info().rss / (1024**3)
    memory_reduction_gb = initial_memory_gb - final_memory_gb

    logger.debug(
        f"Iterative RSSO completed: {n_iterations} iterations, final RMS: {final_convergence_rms:.3f} nm"
    )
    logger.debug(
        f"Final memory usage: {final_memory_gb:.2f} GB (change: {memory_reduction_gb:+.2f} GB)"
    )

    # Apply final cumulative drift corrections to the dataset
    locs["x"] = original_locs["x"] + cumulative_corrections_x
    locs["y"] = original_locs["y"] + cumulative_corrections_y

    # Store drift trajectory for plotting
    drift = np.column_stack([drift_x, drift_y])

    # Store comprehensive results
    results["success"] = True
    results["n_iterations"] = n_iterations
    results["convergence_rms"] = final_convergence_rms
    results["converged"] = final_convergence_rms < convergence_threshold

    # Drift statistics
    drift_magnitude_x = np.max(np.abs(drift_x))
    drift_magnitude_y = np.max(np.abs(drift_y))
    total_drift = np.sqrt(drift_magnitude_x**2 + drift_magnitude_y**2)
    mean_drift_quality = np.mean(drift_quality[drift_quality > 0])

    results["drift_magnitude_x"] = drift_magnitude_x
    results["drift_magnitude_y"] = drift_magnitude_y
    results["total_drift"] = total_drift
    results["mean_drift_quality"] = mean_drift_quality

    # Store subsampling performance statistics
    results["subsampling_fraction"] = subsampling_fraction
    results["progressive_subsampling"] = progressive_subsampling
    results["final_iteration_full_dataset"] = final_iteration_full_dataset
    if progressive_subsampling:
        results["progressive_subsampling_schedule"] = (
            progressive_subsampling_schedule
        )
        # Calculate average speedup across iterations
        avg_speedup = np.mean(
            [
                1.0 / max(0.01, frac)
                for frac in progressive_subsampling_schedule[:n_iterations]
            ]
        )
        results["estimated_avg_speedup"] = avg_speedup
    else:
        results["estimated_speedup"] = (
            1.0 / subsampling_fraction if subsampling_fraction < 1.0 else 1.0
        )
    results["uncertainty_estimation_enabled"] = enable_uncertainty_estimation
    if enable_uncertainty_estimation:
        valid_uncertainties = uncertainty_x[uncertainty_x > 0]
        if len(valid_uncertainties) > 0:
            results["mean_subsampling_uncertainty_nm"] = np.mean(
                valid_uncertainties
            )
            results["max_subsampling_uncertainty_nm"] = np.max(
                valid_uncertainties
            )
        else:
            results["mean_subsampling_uncertainty_nm"] = 0.0
            results["max_subsampling_uncertainty_nm"] = 0.0

    results["uncertainty_x-mean"] = np.mean(uncertainty_x)
    results["uncertainty_y-mean"] = np.mean(uncertainty_y)
    # Store drift trajectories and uncertainties
    results["drift_x"] = drift_x
    results["drift_y"] = drift_y
    results["uncertainty_x"] = list(uncertainty_x)
    results["uncertainty_y"] = list(uncertainty_y)
    results["confidence"] = confidence
    results["drift_quality"] = drift_quality

    # Store iteration history
    results["iteration_history"] = iteration_history

    logger.debug(
        f"Final drift: X={drift_magnitude_x:.1f} nm, Y={drift_magnitude_y:.1f} nm, Total={total_drift:.1f} nm"
    )
    logger.debug(f"Mean quality: {mean_drift_quality:.2f}")

    # Create drift plots with confidence intervals
    if plot_drift:
        logger.debug("Creating drift plots...")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Create comprehensive drift plots showing all iterations
        fig = plt.figure(figsize=(15, 12))

        # Create subplot layout: 2x2 grid
        ax1 = plt.subplot(2, 2, 1)  # X drift with intermediate iterations
        ax2 = plt.subplot(2, 2, 2)  # Y drift with intermediate iterations
        ax3 = plt.subplot(2, 2, 3)  # X drift final with confidence
        ax4 = plt.subplot(2, 2, 4)  # Y drift final with confidence

        frame_indices = np.arange(n_frames)

        # Define colors for iterations (gradient from light to dark)
        if len(iteration_history) > 1:
            colors_x = plt.cm.Blues(
                np.linspace(0.3, 1.0, len(iteration_history))
            )
            colors_y = plt.cm.Reds(
                np.linspace(0.3, 1.0, len(iteration_history))
            )
        else:
            colors_x = ["blue"]
            colors_y = ["red"]

        # Plot 1: X drift - all iterations
        for i, history in enumerate(iteration_history):
            alpha = (
                0.4 if i < len(iteration_history) - 1 else 1.0
            )  # Final iteration is solid
            linewidth = 1.0 if i < len(iteration_history) - 1 else 2.0
            label = (
                f"Iteration {history['iteration']}"
                if len(iteration_history) > 1
                else "X drift"
            )

            ax1.plot(
                frame_indices,
                history["drift_x"],
                color=colors_x[i],
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )

        ax1.set_ylabel("X Drift (nm)")
        ax1.set_title(f"X Drift Evolution ({n_iterations} iterations)")
        ax1.grid(True, alpha=0.3)
        if len(iteration_history) > 1:
            ax1.legend(fontsize=8, loc="best")

        # Plot 2: Y drift - all iterations
        for i, history in enumerate(iteration_history):
            alpha = 0.4 if i < len(iteration_history) - 1 else 1.0
            linewidth = 1.0 if i < len(iteration_history) - 1 else 2.0
            label = (
                f"Iteration {history['iteration']}"
                if len(iteration_history) > 1
                else "Y drift"
            )

            ax2.plot(
                frame_indices,
                history["drift_y"],
                color=colors_y[i],
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )

        ax2.set_ylabel("Y Drift (nm)")
        ax2.set_title(f"Y Drift Evolution ({n_iterations} iterations)")
        ax2.grid(True, alpha=0.3)
        if len(iteration_history) > 1:
            ax2.legend(fontsize=8, loc="best")

        # Plot 3: Final X drift with confidence intervals
        ax3.plot(
            frame_indices, drift_x, "b-", linewidth=2, label="Final X drift"
        )

        # Add confidence intervals if available
        if not np.all(np.isnan(uncertainty_x)) and np.any(uncertainty_x > 0):
            ax3.fill_between(
                frame_indices,
                drift_x - uncertainty_x,
                drift_x + uncertainty_x,
                alpha=0.3,
                color="blue",
                label="±1σ uncertainty",
            )

            # Add 2σ confidence interval if uncertainty is meaningful
            ax3.fill_between(
                frame_indices,
                drift_x - 2 * uncertainty_x,
                drift_x + 2 * uncertainty_x,
                alpha=0.15,
                color="blue",
                label="±2σ confidence",
            )

        ax3.set_xlabel("Frame")
        ax3.set_ylabel("X Drift (nm)")
        ax3.set_title("Final X Drift with Confidence Intervals")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        # Plot 4: Final Y drift with confidence intervals
        ax4.plot(
            frame_indices, drift_y, "r-", linewidth=2, label="Final Y drift"
        )

        # Add confidence intervals if available
        if not np.all(np.isnan(uncertainty_y)) and np.any(uncertainty_y > 0):
            ax4.fill_between(
                frame_indices,
                drift_y - uncertainty_y,
                drift_y + uncertainty_y,
                alpha=0.3,
                color="red",
                label="±1σ uncertainty",
            )

            # Add 2σ confidence interval if uncertainty is meaningful
            ax4.fill_between(
                frame_indices,
                drift_y - 2 * uncertainty_y,
                drift_y + 2 * uncertainty_y,
                alpha=0.15,
                color="red",
                label="±2σ confidence",
            )

        ax4.set_xlabel("Frame")
        ax4.set_ylabel("Y Drift (nm)")
        ax4.set_title("Final Y Drift with Confidence Intervals")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)

        plt.tight_layout()

        # Save drift plot with random code for unique filename
        import random
        import string

        rcode = "".join(random.choices(string.ascii_letters, k=6))
        drift_plot_path = os.path.join(
            results_folder, f"drift_rsso_iterative_{rcode}.png"
        )
        plt.savefig(drift_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        results["drift_plot"] = drift_plot_path

        # Create convergence and statistics plots if multiple iterations
        if n_iterations > 1:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=(14, 10)
            )

            iterations = [h["iteration"] for h in iteration_history]
            rms_values = [h["convergence_rms"] for h in iteration_history]
            valid_measurements = [
                h["valid_measurements"] for h in iteration_history
            ]

            # Plot 1: RMS Convergence
            ax1.plot(
                iterations,
                rms_values,
                "g-o",
                linewidth=2,
                markersize=8,
                label="RMS Change",
            )
            ax1.axhline(
                y=convergence_threshold,
                color="r",
                linestyle="--",
                label=f"Convergence threshold ({convergence_threshold:.3f} nm)",
            )
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("RMS Change (nm)")
            ax1.set_title("RSSO Convergence History")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_yscale("log")  # Log scale often better for convergence

            # Plot 2: Valid measurements per iteration
            all_iterations = [h["iteration"] for h in iteration_history]
            ax2.plot(
                all_iterations,
                valid_measurements,
                "b-s",
                linewidth=2,
                markersize=6,
            )
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Valid Measurements")
            ax2.set_title("Valid Frame Measurements per Iteration")
            ax2.grid(True, alpha=0.3)

            # Plot 3: Mean uncertainty evolution
            mean_uncertainty_x = []
            mean_uncertainty_y = []
            mean_sigma_x = []
            mean_sigma_y = []
            for hist in iteration_history:
                unc_x = hist.get("uncertainty_x", np.array([]))
                unc_y = hist.get("uncertainty_y", np.array([]))
                if len(unc_x) > 0 and not np.all(np.isnan(unc_x)):
                    mean_uncertainty_x.append(np.nanmean(unc_x))
                else:
                    mean_uncertainty_x.append(np.nan)
                if len(unc_y) > 0 and not np.all(np.isnan(unc_y)):
                    mean_uncertainty_y.append(np.nanmean(unc_y))
                else:
                    mean_uncertainty_y.append(np.nan)

                sig_x = hist.get("sigma_x", np.array([]))
                sig_y = hist.get("sigma_y", np.array([]))
                if len(sig_x) > 0 and not np.all(np.isnan(sig_x)):
                    mean_sigma_x.append(np.nanmean(sig_x))
                else:
                    mean_sigma_x.append(np.nan)
                if len(sig_y) > 0 and not np.all(np.isnan(sig_y)):
                    mean_sigma_y.append(np.nanmean(sig_y))
                else:
                    mean_sigma_y.append(np.nan)

            line_ux = ax3.plot(
                all_iterations,
                mean_uncertainty_x,
                "b-o",
                label="X fit uncertainty",
                linewidth=2,
                markersize=6,
            )
            line_uy = ax3.plot(
                all_iterations,
                mean_uncertainty_y,
                "r-o",
                label="Y fit uncertainty",
                linewidth=2,
                markersize=6,
            )
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Mean Uncertainty from fit (nm)")
            ax3_1 = ax3.twinx()
            line_sx = ax3_1.plot(
                all_iterations,
                mean_sigma_x,
                "b:x",
                label="X RSSO sigma",
                linewidth=2,
                markersize=6,
            )
            line_sy = ax3_1.plot(
                all_iterations,
                mean_sigma_y,
                "r:x",
                label="Y RSSO sigma",
                linewidth=2,
                markersize=6,
            )
            ax3_1.set_ylabel("Mean RSSO Sigma (nm)")
            ax3.set_title("Uncertainty Evolution")
            # ax3.set_yscale("log")
            ax3.set_ylim(bottom=0)
            ax3_1.set_ylim(bottom=0)
            ax3.grid(True, alpha=0.3)
            lines = line_ux + line_uy + line_sx + line_sy
            labels = [line.get_label() for line in lines]
            ax3.legend(lines, labels)

            # Plot 4: Mean confidence evolution
            mean_confidence = []
            for hist in iteration_history:
                conf = hist.get("confidence", np.array([]))
                if len(conf) > 0 and not np.all(np.isnan(conf)):
                    mean_confidence.append(np.nanmean(conf))
                else:
                    mean_confidence.append(np.nan)

            ax4.plot(
                all_iterations,
                mean_confidence,
                "purple",
                marker="D",
                linewidth=2,
                markersize=6,
            )
            ax4.set_xlabel("Iteration")
            ax4.set_ylabel("Mean Confidence")
            ax4.set_title("Confidence Evolution")
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)

            plt.tight_layout()

            rcode = "".join(random.choices(string.ascii_letters, k=6))
            convergence_plot_path = os.path.join(
                results_folder, f"convergence_rsso_iterative_{rcode}.png"
            )
            plt.savefig(convergence_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            results["convergence_plot"] = convergence_plot_path

            # Create a summary statistics plot for robustness assessment
            fig, ax = plt.subplots(figsize=(12, 8))

            # Show drift range evolution for robustness check
            drift_ranges_x = []
            drift_ranges_y = []
            for hist in iteration_history:
                drift_x_hist = hist["drift_x"]
                drift_y_hist = hist["drift_y"]
                # Calculate 95th percentile range (robust measure)
                range_x = np.percentile(drift_x_hist, 95) - np.percentile(
                    drift_x_hist, 5
                )
                range_y = np.percentile(drift_y_hist, 95) - np.percentile(
                    drift_y_hist, 5
                )
                drift_ranges_x.append(range_x)
                drift_ranges_y.append(range_y)

            ax.plot(
                all_iterations,
                drift_ranges_x,
                "b-o",
                label="X drift range (90%ile)",
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                all_iterations,
                drift_ranges_y,
                "r-o",
                label="Y drift range (90%ile)",
                linewidth=2,
                markersize=6,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Drift Range (nm)")
            ax.set_title("Drift Range Evolution (Robustness Assessment)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            rcode = "".join(random.choices(string.ascii_letters, k=6))
            robustness_plot_path = os.path.join(
                results_folder, f"robustness_rsso_iterative_{rcode}.png"
            )
            plt.savefig(robustness_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            results["robustness_plot"] = robustness_plot_path

        # Print summary of generated plots
        print("Generated drift analysis plots:")
        print(f"  - Main drift plot: {drift_plot_path}")
        if n_iterations > 1:
            print(f"  - Convergence analysis: {convergence_plot_path}")
            print(f"  - Robustness assessment: {robustness_plot_path}")
            print("  Plot features:")
            print("    • Intermediate iterations shown with color gradients")
            print(
                "    • Final iteration with ±1σ and ±2σ confidence intervals"
            )
            print("    • Convergence history on log scale")
            print("    • Uncertainty and confidence evolution")
            print("    • Drift range evolution for robustness assessment")

    # Save final undrifted localizations
    if save_locs:
        fp_locs = os.path.join(
            results_folder, "locs_undrifted_rsso_iterative.hdf5"
        )
        io.save_locs(fp_locs, locs, info)
        results["fp_locs"] = fp_locs
    return locs, drift, results
