#!/usr/bin/env python3
"""Read-only preflight for a logical-ID Stacking replay specification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def load_json(path: Path | None, encoded: str | None) -> dict[str, Any]:
    if (path is None) == (encoded is None):
        raise ValueError("provide exactly one path or base64-encoded JSON value")
    if path is not None:
        return json.loads(path.read_text())
    assert encoded is not None
    return json.loads(base64.b64decode(encoded).decode("utf-8"))


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **details})


def _is_inside(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_spec(spec: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "estimator_contract",
        "state",
        "source_script",
        "analysis",
        "inputs",
        "expected_outputs",
        "output_namespace",
    }
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"specification is missing keys: {sorted(missing)}")
    if spec["schema_version"] != 1:
        raise ValueError("unsupported schema version")
    if spec["estimator_contract"] != "legacy_reproduction":
        raise ValueError("this preflight accepts only legacy_reproduction")
    if spec["state"] != "planned":
        raise ValueError("preflight specification must be planned")
    logical_ids = [item["logical_id"] for item in spec["inputs"]]
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("input logical IDs must be unique")


def run_preflight(
    spec: dict[str, Any],
    path_map: dict[str, Any],
    *,
    hash_mode: str = "full",
) -> dict[str, Any]:
    validate_spec(spec)
    checks: list[dict[str, Any]] = []
    mapped_inputs = path_map.get("inputs", {})

    expected_ids = {item["logical_id"] for item in spec["inputs"]}
    add_check(
        checks,
        "logical_input_mapping_complete",
        set(mapped_inputs) == expected_ids,
        expected_count=len(expected_ids),
        mapped_count=len(mapped_inputs),
    )

    source_path = Path(path_map.get("source_script", ""))
    source_exists = source_path.is_file()
    source_hash = sha256(source_path) if source_exists and hash_mode == "full" else None
    add_check(
        checks,
        "historical_source_script",
        source_exists
        and (hash_mode != "full" or source_hash == spec["source_script"]["sha256"]),
        logical_id=spec["source_script"]["logical_id"],
        observed_sha256=source_hash,
    )

    for artifact in spec["inputs"]:
        logical_id = artifact["logical_id"]
        raw_path = mapped_inputs.get(logical_id)
        if raw_path is None:
            add_check(checks, "input_artifact", False, logical_id=logical_id, reason="unmapped")
            continue
        path = Path(raw_path)
        if not path.is_file():
            add_check(checks, "input_artifact", False, logical_id=logical_id, reason="missing")
            continue
        observed_bytes = path.stat().st_size
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            observed_shape = list(array.shape)
            observed_dtype = str(array.dtype)
        except Exception as error:  # report compactly; do not expose private paths
            add_check(
                checks,
                "input_artifact",
                False,
                logical_id=logical_id,
                reason=f"header_error:{type(error).__name__}",
            )
            continue
        observed_hash = sha256(path) if hash_mode == "full" else None
        passed = (
            observed_bytes == artifact["bytes"]
            and observed_shape == artifact["shape"]
            and observed_dtype == artifact["dtype"]
            and (hash_mode != "full" or observed_hash == artifact["sha256"])
        )
        add_check(
            checks,
            "input_artifact",
            passed,
            logical_id=logical_id,
            observed_bytes=observed_bytes,
            observed_shape=observed_shape,
            observed_dtype=observed_dtype,
            observed_sha256=observed_hash,
        )

    historical_output = Path(path_map.get("historical_output_directory", "")).resolve()
    proposed_output = Path(path_map.get("proposed_output_directory", "")).resolve()
    output_safe = (
        str(path_map.get("proposed_output_directory", "")) != ""
        and proposed_output != historical_output
        and not _is_inside(proposed_output, historical_output)
        and not proposed_output.exists()
    )
    add_check(
        checks,
        "fresh_output_namespace",
        output_safe,
        proposed_exists=proposed_output.exists(),
        separated_from_historical=not _is_inside(proposed_output, historical_output),
    )

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "status": "passed" if passed else "failed",
        "hash_mode": hash_mode,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": module_version("scipy"),
            "sklearn": module_version("sklearn"),
            "cvxopt": module_version("cvxopt"),
        },
        "rng_gate": {
            "classification": spec["analysis"]["pca_rng_classification"],
            "pca_random_state": spec["analysis"]["pca_random_state"],
            "launch_allowed": spec["analysis"]["pca_rng_classification"] != "unresolved_historical",
        },
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    spec_group = parser.add_mutually_exclusive_group(required=True)
    spec_group.add_argument("--spec", type=Path)
    spec_group.add_argument("--spec-base64")
    map_group = parser.add_mutually_exclusive_group(required=True)
    map_group.add_argument("--path-map", type=Path)
    map_group.add_argument("--path-map-base64")
    parser.add_argument("--hash-mode", choices=["full", "header-only"], default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        spec = load_json(args.spec, args.spec_base64)
        path_map = load_json(args.path_map, args.path_map_base64)
        result = run_preflight(spec, path_map, hash_mode=args.hash_mode)
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "fatal_error": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
