#!/usr/bin/env python3
"""Atomically attach approved ELF3 Sonic step28800 motion tokens to one episode.

This tool intentionally does not implement or approximate the encoder.  The
customer encoder produces a (T, 64) NumPy file; this command verifies its
alignment and FSQ contract, records provenance, and installs it as a sidecar
used by the strict LeRobot v2.1 exporter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from real_robot_collection.contracts import MOTION_TOKEN_DIM
from validate_raw import read_rows


TOKEN_FILENAME = "action_motion_token.npy"
MANIFEST_FILENAME = "motion_token.json"
TOKEN_SOURCE = "customer_sonic_step28800_post_fsq_decoder_input"


def _validated_sha256(value: str, label: str) -> str:
    text = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_token_array(value, expected_rows: int) -> np.ndarray:
    token = np.asarray(value, dtype=np.float64)
    expected = (int(expected_rows), MOTION_TOKEN_DIM)
    if token.shape != expected:
        raise ValueError(f"motion tokens must have shape {expected}, got {token.shape}")
    if not np.isfinite(token).all():
        raise ValueError("motion tokens contain NaN or Inf")
    scaled = token * 16.0
    if not np.allclose(scaled, np.rint(scaled), rtol=0.0, atol=1.0e-8):
        raise ValueError("motion tokens are not on the expected 1/16 FSQ grid")
    return np.ascontiguousarray(token, dtype=np.float64)


def attach_tokens(
    episode: Path,
    token_file: Path,
    *,
    encoder_checkpoint_sha256: str,
    decoder_checkpoint_sha256: str,
) -> dict:
    episode = episode.expanduser().resolve()
    if not (episode / "meta.json").is_file():
        raise ValueError(f"not a source episode: {episode}")
    rows = read_rows(episode / "frames.jsonl")
    with token_file.expanduser().resolve().open("rb") as handle:
        raw = np.load(handle, allow_pickle=False)
    token = validate_token_array(raw, len(rows))
    encoder_checkpoint_sha256 = _validated_sha256(
        encoder_checkpoint_sha256, "encoder checkpoint SHA-256"
    )
    decoder_checkpoint_sha256 = _validated_sha256(
        decoder_checkpoint_sha256, "decoder checkpoint SHA-256"
    )
    destination = episode / TOKEN_FILENAME
    manifest_path = episode / MANIFEST_FILENAME
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(
            f"token sidecar already exists in {episode}; create a new dataset version "
            "instead of silently replacing training data"
        )
    tmp = episode / f".{TOKEN_FILENAME}.tmp"
    with tmp.open("wb") as handle:
        np.save(handle, token, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    token_sha256 = sha256_file(tmp)
    manifest = {
        "source": TOKEN_SOURCE,
        "dimension": MOTION_TOKEN_DIM,
        "dtype": "float64",
        "fsq_levels": 32,
        "fsq_grid": "1/16",
        "frames": len(rows),
        "encoder_checkpoint_sha256": str(encoder_checkpoint_sha256),
        "decoder_checkpoint_sha256": str(decoder_checkpoint_sha256),
        "token_file": TOKEN_FILENAME,
        "token_file_sha256": token_sha256,
    }
    manifest_tmp = episode / f".{MANIFEST_FILENAME}.tmp"
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    os.replace(manifest_tmp, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="attach an approved (T,64) ELF3 step28800 token sidecar"
    )
    parser.add_argument("episode", type=Path)
    parser.add_argument("tokens", type=Path, help="NumPy .npy array with shape (T,64)")
    parser.add_argument("--encoder-checkpoint-sha256", required=True)
    parser.add_argument("--decoder-checkpoint-sha256", required=True)
    args = parser.parse_args()
    try:
        manifest = attach_tokens(
            args.episode,
            args.tokens,
            encoder_checkpoint_sha256=args.encoder_checkpoint_sha256,
            decoder_checkpoint_sha256=args.decoder_checkpoint_sha256,
        )
    except Exception as exc:
        print(f"TOKEN ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"TOKEN OK: frames={manifest['frames']} dimension={manifest['dimension']} "
        f"source={manifest['source']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
