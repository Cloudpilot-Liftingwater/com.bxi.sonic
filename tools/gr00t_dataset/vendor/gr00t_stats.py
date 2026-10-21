# SPDX-License-Identifier: Apache-2.0
# NVIDIA upstream absolute-statistics subset; function bodies unchanged.
import hashlib,json,logging,os,tempfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
logger=logging.getLogger(__name__)
LE_ROBOT_DATA_FILENAME="data/*/*.parquet"
LE_ROBOT_INFO_FILENAME="meta/info.json"
LE_ROBOT_STATS_FILENAME="meta/stats.json"
STATS_FINGERPRINTS_KEY="__fingerprints__"


def _load_stats_cache(path: Path) -> dict[str, Any]:
    """Load a stats JSON cache, treating any unreadable state as "no cache".

    A stats file becomes unreadable when a previous writer was killed mid-flush
    (ENOSPC, SIGKILL, runner reboot) and left a 0-byte / truncated file behind.
    Without this guard, the leftover file traps every subsequent caller in a
    ``json.JSONDecodeError`` until a human deletes it — observed taking down 6
    of 8 retried test.unit.gpu jobs after a /shared NFS ENOSPC event.

    Empty file, missing file, JSON parse error, and OSError are all treated
    equivalently: regenerate from scratch. Callers MUST then write back via
    :func:`_dump_stats_cache_atomic` so the same partial-write scenario does
    not recur on the very next ENOSPC.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return {}
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[stats] discarding unreadable cache %s: %s; regenerating", path, exc)
        return {}

def _dump_stats_cache_atomic(path: Path, data: dict[str, Any], *, indent: int | None = 4) -> None:
    """Atomically write *data* as JSON to *path* via tmp-file + ``os.replace``.

    Avoids the leave-a-0-byte-file failure mode that motivates
    :func:`_load_stats_cache`: ``open(path, "w")`` truncates immediately, so a
    SIGKILL between truncate and the final ``write()`` poisons the cache for
    every future caller. Writing to a unique sibling temp file and then
    ``os.replace`` guarantees that *path* either points to the previous valid
    content (writer killed) or to the new fully-flushed content (writer
    succeeded) — never to a partial intermediate.

    The temp filename must be unique per writer. CI can run multiple GPU jobs
    against the same cached dataset path, and a fixed ``<name>.tmp`` lets one
    writer rename or clean up another writer's temp file.

    Best-effort cleanup of the tmp file on exception so we don't litter
    ``meta/`` with abandoned ``*.tmp`` shards.

    NFS durability: explicitly ``flush`` + ``fsync`` before ``os.replace`` so
    the tmp file's bytes are forced from the page cache to the storage
    backend before the rename makes the new name visible. Without this, a
    SIGKILL between a successful ``os.replace`` and the kernel's writeback
    can still leave a 0-byte file on NFS after a client reconnect — the
    very failure mode this helper exists to prevent.
    """
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp = Path(f.name)
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics

def _compute_stats_fingerprint(feature_name: str, feature_meta: dict) -> str:
    """Hash the per-feature schema in ``info.json`` that drives ``calculate_dataset_statistics``.

    Without this, ``meta/stats.json`` was reused whenever every feature name was
    still present, even if the underlying ``dtype`` / ``shape`` had changed
    (e.g. column dim grew, dtype widened). Result: silently wrong normalization
    at training/eval time. Hashing the per-feature schema makes any such change
    invalidate just that feature's cached entry.
    """
    payload = {
        "feature": feature_name,
        "dtype": feature_meta.get("dtype"),
        "shape": feature_meta.get("shape"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _stale_features(stats: dict | None, le_features: dict, lowdim_features: list[str]) -> list[str]:
    """Return the subset of ``lowdim_features`` whose cached entry is missing or stale.

    A feature is considered fresh iff its stat-dict has all six fields and its
    fingerprint in ``__fingerprints__`` matches the canonical hash for its
    current ``info.json`` schema. Anything else (missing info entry, missing
    stat entry, missing stat field, missing fingerprint, mismatched
    fingerprint) is treated as stale and recomputed.
    """
    if stats is None:
        return list(lowdim_features)
    fingerprints = stats.get(STATS_FINGERPRINTS_KEY)
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    stale = []
    for feature in lowdim_features:
        feature_meta = le_features.get(feature)
        if feature_meta is None:
            stale.append(feature)
            continue
        if feature not in stats or not isinstance(stats[feature], dict):
            stale.append(feature)
            continue
        if any(k not in stats[feature] for k in ("mean", "std", "min", "max", "q01", "q99")):
            stale.append(feature)
            continue
        if fingerprints.get(feature) != _compute_stats_fingerprint(feature, feature_meta):
            stale.append(feature)
    return stale

def check_stats_validity(dataset_path: Path | str, features: list[str]):
    """Return True iff every feature in ``features`` has a fingerprint-matching cached entry.

    A True result means ``generate_stats`` can skip recomputation entirely. We
    re-derive the expected fingerprint from the *current* ``info.json`` so any
    schema drift since the cache was written invalidates it.
    """
    dataset_path = Path(dataset_path)
    stats = _load_stats_cache(dataset_path / LE_ROBOT_STATS_FILENAME)
    if not stats:
        return False
    info_path = dataset_path / LE_ROBOT_INFO_FILENAME
    if not info_path.exists():
        return False
    with open(info_path, "r") as f:
        le_features = json.load(f).get("features", {})
    return not _stale_features(stats, le_features, features)

def generate_stats(dataset_path: Path | str):
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {str(dataset_path)}")
    with open(dataset_path / LE_ROBOT_INFO_FILENAME, "r") as f:
        le_features = json.load(f)["features"]
    lowdim_features = [f for f in le_features if "float" in le_features[f]["dtype"]]

    stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
    existing = _load_stats_cache(stats_path)
    stale = _stale_features(existing, le_features, lowdim_features)

    # Pull the reserved sidecar aside so the cleanup pass below can iterate
    # ``existing`` cleanly. Drop entries for features that no longer exist in
    # info.json (e.g. a sensor / DOF was removed in an upstream dataset rev)
    # so the on-disk file stays consistent with info.json under feature
    # churn — otherwise stale stat dicts accumulate without bound on shared
    # NFS. Any non-stat-dict, non-sidecar key is owned by an external writer
    # and is left untouched.
    fingerprints = existing.pop(STATS_FINGERPRINTS_KEY, None)
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    lowdim_set = set(lowdim_features)
    dropped = False
    for f in [k for k in list(existing) if k not in lowdim_set and isinstance(existing[k], dict)]:
        del existing[f]
        dropped = True
    for f in [k for k in list(fingerprints) if k not in lowdim_set]:
        del fingerprints[f]
        dropped = True

    if not stale and not dropped:
        return

    parquet_files = list(dataset_path.glob(LE_ROBOT_DATA_FILENAME))
    fresh = calculate_dataset_statistics(parquet_files, stale) if stale else {}
    for feature, values in fresh.items():
        existing[feature] = values
        fingerprints[feature] = _compute_stats_fingerprint(feature, le_features[feature])

    existing[STATS_FINGERPRINTS_KEY] = fingerprints
    _dump_stats_cache_atomic(stats_path, existing)
