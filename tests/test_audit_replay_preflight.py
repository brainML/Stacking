import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_replay_preflight import run_preflight


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(tmp_path: Path):
    source = tmp_path / "legacy.py"
    source.write_text("print('legacy')\n")
    array_path = tmp_path / "input.npy"
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(array_path, array)
    historical = tmp_path / "historical"
    historical.mkdir()
    proposed = tmp_path / "new-output"
    logical_id = "fixture/input"
    spec = {
        "schema_version": 1,
        "run_id": "fixture",
        "estimator_contract": "legacy_reproduction",
        "state": "planned",
        "source_script": {"logical_id": "fixture/source", "sha256": file_hash(source)},
        "analysis": {
            "pca_rng_classification": "prospectively_seeded_reconstruction",
            "pca_random_state": 0,
        },
        "inputs": [
            {
                "logical_id": logical_id,
                "shape": [3, 4],
                "dtype": "float32",
                "bytes": array_path.stat().st_size,
                "sha256": file_hash(array_path),
            }
        ],
        "expected_outputs": [],
        "output_namespace": "fixture/output",
    }
    path_map = {
        "inputs": {logical_id: str(array_path)},
        "source_script": str(source),
        "historical_output_directory": str(historical),
        "proposed_output_directory": str(proposed),
    }
    return spec, path_map, historical


def test_full_preflight_passes_without_creating_output(tmp_path):
    spec, path_map, _ = make_fixture(tmp_path)

    result = run_preflight(spec, path_map, hash_mode="full")

    assert result["status"] == "passed"
    assert result["rng_gate"]["launch_allowed"] is True
    assert not Path(path_map["proposed_output_directory"]).exists()


def test_hash_mismatch_fails(tmp_path):
    spec, path_map, _ = make_fixture(tmp_path)
    spec = json.loads(json.dumps(spec))
    spec["inputs"][0]["sha256"] = "0" * 64

    result = run_preflight(spec, path_map, hash_mode="full")

    assert result["status"] == "failed"
    artifact_check = next(check for check in result["checks"] if check["name"] == "input_artifact")
    assert artifact_check["passed"] is False


def test_output_inside_historical_directory_fails(tmp_path):
    spec, path_map, historical = make_fixture(tmp_path)
    path_map["proposed_output_directory"] = str(historical / "overwrite-risk")

    result = run_preflight(spec, path_map, hash_mode="header-only")

    assert result["status"] == "failed"
    namespace_check = next(
        check for check in result["checks"] if check["name"] == "fresh_output_namespace"
    )
    assert namespace_check["passed"] is False
