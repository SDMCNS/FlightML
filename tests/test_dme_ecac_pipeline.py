from pathlib import Path
import pytest
from typer.testing import CliRunner

from flightml.data.dataset import FlightDataset
from flightml.eval.dme_ecac_pipeline import (
    get_ranked_ecac_firs,
    record_network_evaluation,
    load_network_evaluations,
    run_ecac_dme_pipeline,
)
from flightml.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def dataset():
    return FlightDataset(period="202510")


def test_get_ranked_ecac_firs_descending(dataset):
    """Verify that ECAC FIRs are returned ranked in descending order of traffic moves."""
    ranked = get_ranked_ecac_firs(dataset, period="202510")
    assert len(ranked) >= 5
    for i in range(len(ranked) - 1):
        assert ranked[i]["traffic_moves"] >= ranked[i + 1]["traffic_moves"]
    assert ranked[0]["dme_count"] > 0


def test_record_and_load_network_evaluations(tmp_path):
    """Verify writing and reading from persistent network evaluations dataset."""
    test_jsonl = tmp_path / "test_evals.jsonl"
    rec1 = {
        "period": "202510",
        "target_standard": "RNAV_5",
        "fir_id": "EGTTFIR",
        "fir_name": "London FIR",
        "rank_by_traffic": 1,
        "traffic_monthly_moves": 180000,
        "baseline_dmes": 43,
        "optimal_active_dmes": 16,
        "decommissioned_dmes": 27,
        "equipment_reduction_pct": 62.8,
        "coverage_compliance_pct": 96.0,
        "annual_savings_eur": 1215000.0,
    }
    record_network_evaluation(rec1, path=test_jsonl)
    df = load_network_evaluations(path=test_jsonl)
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]["fir_id"] == "EGTTFIR"
    assert df.iloc[0]["annual_savings_eur"] == 1215000.0


def test_run_ecac_dme_pipeline_subset(dataset):
    """Verify running the end-to-end ECAC pipeline for top 2 busiest FIRs."""
    res = run_ecac_dme_pipeline(
        dataset=dataset,
        period="202510",
        target_standard="RNAV_5",
        max_firs=2,
        max_permutations_per_fir=3,
    )
    assert res["total_firs_evaluated"] == 2
    assert Path(res["summary_report_path"]).exists()
    assert Path(res["summary_report_path"]).stat().st_size > 20_000
    assert len(res["fir_report_paths"]) == 2


def test_cli_dme_ecac_pipeline():
    """Verify CLI command `flightml dme-ecac-pipeline` execution on top 2 FIRs."""
    result = runner.invoke(
        app,
        [
            "dme-ecac-pipeline",
            "--period", "202510",
            "--standard", "RNAV_5",
            "--max-firs", "2",
            "--max-permutations", "3",
        ],
    )
    assert result.exit_code == 0
    assert "Pan-European ECAC Resilience & Rationalization Pipeline" in result.output
    assert "ECAC Sovereign FIRs Sequenced by Traffic Density" in result.output
    assert "Pan-European Evaluation Complete" in result.output
    assert "ECAC Resilience Pipeline Execution Complete" in result.output
