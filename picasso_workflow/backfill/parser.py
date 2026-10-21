"""Parse a FLAT archive dataset into registry features + a stable ``run_id``.

The pre-automation archive is untyped: a ``*_locs.hdf5`` localization file, its
``.yaml`` info sidecar, and (near it) the raw movie, laid out under folders
whose names encode light metadata by lab convention. This module turns one such
dataset into:

* a **content-derived, stable ``run_id``** (a ULID whose randomness bits are a
  SHA-256 of the dataset's identity) so re-running the sweep is idempotent — a
  file whose run already exists is skipped, mirroring
  :mod:`picasso_registry.backfill_liveloc`'s deterministic-id scheme;
* **features** harvested from the ``.yaml`` info blocks and folder-name
  conventions (operator, microscope, frames, exposure, pixel size, FOV
  extent, and any ``key=value`` / ``key-value`` tokens in the path);
* a **``metadata_complete`` gate**: True only when the descriptor-defining
  fields are present. Untouched-default / metadata-incomplete datasets are
  ingested (history is append-only) but EXCLUDED from the learned per-taxon
  reference ranges (:mod:`.ranges`) — the WP-7 "learn only from
  metadata-complete records" rule.

No taxonomy guessing: the sample *descriptor* (taxon / target / modality) is
human knowledge and arrives via the curated CSV (:mod:`.csv_import`), not
inferred from folder names here. The folder tokens are kept verbatim as
features for later curation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Per-table salts so the five FK-joined rows of one dataset get distinct ids
#: from the same identity string (same scheme as backfill_liveloc).
_TABLE_SALTS = ("exp", "chan", "acq", "fov", "ana")

#: Fields that must be present (from folder tokens + yaml) for a dataset to
#: count as metadata-complete and feed the learned per-taxon ranges. These are
#: the cheap, file-derivable completeness signals; the descriptor axes proper
#: are verified via the CSV, which gates taxon assignment separately.
_REQUIRED_FEATURE_KEYS = ("pixelsize_nm", "frames", "exposure_ms")

#: folder-name token ``key=value`` or ``key-value`` (e.g. ``exp-30ms``,
#: ``conc=1nM``). ``_`` is a token *boundary* (not a separator), so an
#: underscore-joined name like ``c0_exp-30ms`` parses as the token ``exp=30ms``,
#: not ``c0=exp-30ms``. Keys are lower-cased; values kept verbatim up to the
#: next boundary (``_`` / ``/``).
_TOKEN_RE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9]*)[=-](?P<val>[^_/-]+)")


def _ulid_cls():
    """Lazy python-ulid import with an actionable error (as backfill_liveloc)."""
    try:
        from ulid import ULID
    except ModuleNotFoundError as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "python-ulid is required for the backfill — install the client "
            "stack: pip install 'picasso-registry[client]'"
        ) from exc
    return ULID


def _info_get(info: Sequence[Mapping[str, Any]], key: str) -> Any:
    """First non-None ``key`` across picasso's list-of-dicts info blocks."""
    for part in info or ():
        val = part.get(key)
        if val is not None:
            return val
    return None


def _coerce_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _coerce_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def parse_folder_tokens(path: Path) -> dict[str, str]:
    """Harvest ``key=value`` / ``key-value`` tokens from the dataset path.

    Kept verbatim (lower-cased keys) as free-form features for later curation;
    NOT interpreted as descriptor axes here. Later path components win on a
    repeated key (the dataset folder is more specific than the project root).
    """
    tokens: dict[str, str] = {}
    for part in path.parts:
        for m in _TOKEN_RE.finditer(part):
            tokens[m.group("key").lower()] = m.group("val")
    return tokens


@dataclass
class DatasetFeatures:
    """Features + identity for one flat-archive ``_locs.hdf5`` dataset."""

    locs_path: str
    yaml_path: str | None
    run_id: str
    #: per-table ids (exp/chan/acq/fov/ana) derived from the same identity.
    ids: dict[str, str]
    #: yaml-derived acquisition features (nm, ms, counts).
    pixelsize_nm: float | None = None
    frames: int | None = None
    exposure_ms: float | None = None
    width_px: int | None = None
    height_px: int | None = None
    operator: str | None = None
    microscope: str | None = None
    created_at: str | None = None
    #: verbatim folder-name tokens (free-form, for later curation).
    folder_tokens: dict[str, str] = field(default_factory=dict)
    #: the full yaml info list, kept verbatim in acquisition_run.extra.
    raw_info: list[dict] = field(default_factory=list)
    metadata_complete: bool = False
    metadata_missing: list[str] = field(default_factory=list)


def _identity(locs_path: str, info: Sequence[Mapping[str, Any]]) -> str:
    """A content-derived identity for the dataset.

    Prefer stable *content* signals (localize step's ``Frames`` + the relative
    file name) over the mount path, so copies under different mounts/backups
    dedupe to one run. Falls back to the file stem when info is bare.
    """
    frames = _info_get(info, "Frames")
    name = Path(locs_path).name
    return f"{name}|frames={frames}"


def _deterministic_ulid(created: datetime, identity: str, salt: str) -> str:
    """ULID: timestamp bits from ``created`` (UTC-pinned), randomness bits a
    SHA-256 of ``identity`` + ``salt`` — stable across re-runs and machines
    (identical scheme to backfill_liveloc, so the dedupe guarantee matches)."""
    ts_ms = int(created.timestamp() * 1000)
    digest = hashlib.sha256(f"{salt}|{identity}".encode()).digest()
    return str(_ulid_cls().from_bytes(ts_ms.to_bytes(6, "big") + digest[:10]))


def _created_dt(info: Sequence[Mapping[str, Any]]) -> tuple[datetime, str]:
    """A ``created`` datetime for id-minting + its iso string.

    Uses a yaml timestamp when present; otherwise a fixed epoch so the id is
    purely content-derived (the timestamp bits then add no entropy but keep the
    ULID well-formed). Naive values are pinned to UTC for cross-machine
    determinism (same reasoning as backfill_liveloc._parse_created)."""
    raw = _info_get(info, "Created") or _info_get(info, "created")
    if raw:
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt, dt.isoformat()
        except ValueError:
            pass
    # No timestamp in the archive file: pin to the epoch so the id depends only
    # on content (identity), keeping re-runs byte-stable.
    return datetime(1970, 1, 1, tzinfo=timezone.utc), ""


def synth_run_id(locs_path: str, info: Sequence[Mapping[str, Any]]) -> str:
    """The stable acquisition ``run_id`` (``ids['acq']``) for a dataset.

    Exposed separately so the idempotency check and tests can mint the id
    without a full parse.
    """
    created, _ = _created_dt(info)
    identity = _identity(locs_path, info)
    return _deterministic_ulid(created, identity, "acq")


def parse_dataset(
    locs_path: str,
    info: Sequence[Mapping[str, Any]],
    *,
    yaml_path: str | None = None,
) -> DatasetFeatures:
    """Parse one dataset's ``.yaml`` info + folder path into features + ids.

    ``info`` is picasso's loaded info list (list of dicts). ``yaml_path`` is the
    ``.yaml`` sidecar path (defaults to the locs path with a ``.yaml``
    extension). The returned :class:`DatasetFeatures` carries the five FK ids,
    the acquisition features, the verbatim folder tokens, and the
    ``metadata_complete`` gate.
    """
    info = list(info or [])
    path = Path(locs_path)
    if yaml_path is None:
        yaml_path = str(path.with_suffix(".yaml"))

    created, created_iso = _created_dt(info)
    identity = _identity(locs_path, info)
    ids = {
        salt: _deterministic_ulid(created, identity, salt)
        for salt in _TABLE_SALTS
    }

    tokens = parse_folder_tokens(path)

    pixelsize_nm = _coerce_float(_info_get(info, "Pixelsize"))
    frames = _coerce_int(_info_get(info, "Frames"))
    width_px = _coerce_int(_info_get(info, "Width"))
    height_px = _coerce_int(_info_get(info, "Height"))
    # Exposure: yaml key varies; also accept an ``exp`` folder token (e.g.
    # ``exp-30ms`` → 30).
    exposure_ms = _coerce_float(
        _info_get(info, "Exposure")
        or _info_get(info, "exposure_ms")
        or _token_number(tokens.get("exp"))
    )
    operator = _info_get(info, "Operator") or tokens.get("user")
    microscope = _info_get(info, "Microscope") or tokens.get("scope")

    feats = DatasetFeatures(
        locs_path=str(locs_path),
        yaml_path=yaml_path,
        run_id=ids["acq"],
        ids=ids,
        pixelsize_nm=pixelsize_nm,
        frames=frames,
        exposure_ms=exposure_ms,
        width_px=width_px,
        height_px=height_px,
        operator=operator,
        microscope=microscope,
        created_at=created_iso or None,
        folder_tokens=tokens,
        raw_info=info,
    )

    missing = [
        key for key in _REQUIRED_FEATURE_KEYS if getattr(feats, key) is None
    ]
    feats.metadata_missing = missing
    feats.metadata_complete = not missing
    return feats


def _token_number(val: str | None) -> float | None:
    """Leading numeric part of a folder token value (``30ms`` → 30.0)."""
    if not val:
        return None
    m = re.match(r"[-+]?\d*\.?\d+", val)
    return float(m.group()) if m else None
