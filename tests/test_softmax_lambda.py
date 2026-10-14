from __future__ import annotations

import json

import numpy as np

from bench.probes.softmax_lambda import compare_outputs, main


def test_softmax_lambda_comparator_reports_hidden_and_prediction_deltas(
    tmp_path,
) -> None:
    reference_hidden = tmp_path / "reference.npy"
    candidate_hidden = tmp_path / "candidate.npy"
    reference_report = tmp_path / "reference.json"
    candidate_report = tmp_path / "candidate.json"
    np.save(reference_hidden, np.zeros((2, 3), dtype=np.float32))
    np.save(
        candidate_hidden,
        np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, -2.0)), dtype=np.float32),
    )
    reference_report.write_text(
        json.dumps(
            {
                "next_token": {"token_id": 7},
                "top_k": [{"token_id": 7}, {"token_id": 8}],
                "performance": {"median_online_seconds": 10.0},
            }
        ),
        encoding="utf-8",
    )
    candidate_report.write_text(
        json.dumps(
            {
                "next_token": {"token_id": 7},
                "top_k": [{"token_id": 7}, {"token_id": 8}],
                "performance": {"median_online_seconds": 9.0},
            }
        ),
        encoding="utf-8",
    )

    result = compare_outputs(
        reference_hidden,
        candidate_hidden,
        reference_report=reference_report,
        candidate_report=candidate_report,
    )

    assert result["shape"] == [2, 3]
    assert result["all"]["max_abs"] == 2.0
    assert result["last_token"]["max_abs"] == 2.0
    assert result["prediction"]["top_k_identical"] is True
    assert result["performance"]["candidate_online_seconds"] == 9.0


def test_softmax_lambda_validation_is_dry_run_by_default(
    tmp_path, capsys
) -> None:
    assert main(["--output-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "--softmax-lambda-bootstrap-iterations 2" in output
    assert "--no-softmax-single-pass-match-two-pass-level" in output
    assert "one_pass_default.json" in output
    assert "dry run only" in output
