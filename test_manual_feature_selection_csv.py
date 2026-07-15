import csv
import json
from pathlib import Path

import pandas as pd
import pytest

import stage2_feature_catalog as catalog


def _ranking_payload():
    ranking = []
    for rank, name in enumerate(catalog.model_candidate_names(), start=1):
        record = catalog.feature_record(name)
        ranking.append({
            "rank": rank,
            "feature": name,
            "eligible_for_manual_selection": True,
            "ineligible_reasons": [],
            "ranking_score": 1.0 / rank,
            "group": record["group"],
            "formula": record["formula"],
            "preprocessing": record["preprocessing"],
            "unit": record["unit"],
            "valid_auc": 0.5,
            "train_group_fold_auc_mean": 0.5,
            "fp_proxy_sample_fp_rate": 0.01,
            "valid_psi": 0.0,
            "deployment_cost": record["deployment_cost"],
            "risk_flags": list(record.get("risk_flags", [])),
        })
    return {
        "schema_version": 1,
        "feature_pool_version": catalog.FEATURE_POOL_VERSION,
        "ranking_policy": {"score_formula": "ranking_score"},
        "ranking": ranking,
    }


def _write_ranking(path: Path):
    path.write_text(json.dumps(_ranking_payload()), encoding="utf-8")


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select(path: Path, names):
    rows = _read_rows(path)
    for row in rows:
        if row["feature"] in names:
            row["selected"] = "1"
    _write_rows(path, rows)


def test_csv_selection_export_has_complete_ordered_contract(tmp_path):
    from manual_feature_selection import CSV_SCHEMA_VERSION, export_manual_selection_csv

    ranking_path = tmp_path / "feature_ranking_full.json"
    _write_ranking(ranking_path)

    output = export_manual_selection_csv(ranking_path, tmp_path)

    assert output == tmp_path / "manual_feature_selection.csv"
    frame = pd.read_csv(output, dtype=str, encoding="utf-8-sig")
    assert frame["feature"].tolist() == catalog.model_candidate_names()
    assert frame["selected"].tolist() == ["0"] * len(catalog.model_candidate_names())
    assert set(frame["csv_schema_version"]) == {str(CSV_SCHEMA_VERSION)}
    assert set(frame["feature_pool_version"]) == {catalog.FEATURE_POOL_VERSION}
    assert frame["ranking_sha256"].nunique() == 1
    assert {"commercial_8_member", "commercial_original_name", "c_operators"} <= set(frame.columns)


def test_csv_selection_import_preserves_catalog_order(tmp_path):
    from manual_feature_selection import export_manual_selection_csv, load_manual_selection_csv

    ranking_path = tmp_path / "feature_ranking_full.json"
    _write_ranking(ranking_path)
    selection_path = export_manual_selection_csv(ranking_path, tmp_path)
    selected = {
        "COMM_GREEN_AC",
        "ACC_MAG_MEAN",
        "GREEN_FFT_PEAK_MEDIAN_RATIO",
        "G_PAIR_PERIODICITY_MEDIAN",
    }
    _select(selection_path, selected)

    actual, provenance = load_manual_selection_csv(
        selection_path,
        ranking_path,
        train_columns=set(catalog.model_candidate_names()),
        valid_columns=set(catalog.model_candidate_names()),
    )

    assert actual == [name for name in catalog.model_candidate_names() if name in selected]
    assert provenance["selection_source_type"] == "csv"
    assert provenance["engineering_warnings"]


def test_csv_selection_rejects_empty_selection_and_stale_ranking(tmp_path):
    from manual_feature_selection import export_manual_selection_csv, load_manual_selection_csv

    ranking_path = tmp_path / "feature_ranking_full.json"
    _write_ranking(ranking_path)
    selection_path = export_manual_selection_csv(ranking_path, tmp_path)

    with pytest.raises(ValueError, match="empty selection"):
        load_manual_selection_csv(
            selection_path, ranking_path,
            train_columns=set(catalog.model_candidate_names()),
            valid_columns=set(catalog.model_candidate_names()),
        )

    _select(selection_path, {catalog.model_candidate_names()[0]})
    payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    payload["ranking"][0]["ranking_score"] = 999.0
    ranking_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ranking SHA256 mismatch"):
        load_manual_selection_csv(
            selection_path, ranking_path,
            train_columns=set(catalog.model_candidate_names()),
            valid_columns=set(catalog.model_candidate_names()),
        )


def test_csv_selection_rejects_changed_immutable_field(tmp_path):
    from manual_feature_selection import export_manual_selection_csv, load_manual_selection_csv

    ranking_path = tmp_path / "feature_ranking_full.json"
    _write_ranking(ranking_path)
    selection_path = export_manual_selection_csv(ranking_path, tmp_path)
    rows = _read_rows(selection_path)
    rows[0]["formula"] = "tampered formula"
    rows[1]["selected"] = "1"
    _write_rows(selection_path, rows)

    with pytest.raises(ValueError, match=r"row 2 field formula.*immutable"):
        load_manual_selection_csv(
            selection_path, ranking_path,
            train_columns=set(catalog.model_candidate_names()),
            valid_columns=set(catalog.model_candidate_names()),
        )


@pytest.mark.parametrize("bad_value", ["yes", "2", "", "1.0"])
def test_csv_selection_rejects_non_binary_selected_values(tmp_path, bad_value):
    from manual_feature_selection import export_manual_selection_csv, load_manual_selection_csv

    ranking_path = tmp_path / "feature_ranking_full.json"
    _write_ranking(ranking_path)
    selection_path = export_manual_selection_csv(ranking_path, tmp_path)
    rows = _read_rows(selection_path)
    rows[0]["selected"] = bad_value
    rows[1]["selected"] = "1"
    _write_rows(selection_path, rows)

    with pytest.raises(ValueError, match="selected must be 0 or 1"):
        load_manual_selection_csv(
            selection_path, ranking_path,
            train_columns=set(catalog.model_candidate_names()),
            valid_columns=set(catalog.model_candidate_names()),
        )


def test_s04_ranking_export_writes_csv_selection_package_only(tmp_path):
    import s04_feature_selection as s04

    outputs = s04.export_full_feature_ranking(tmp_path, _ranking_payload()["ranking"])

    assert outputs["selection_csv"].exists()
    assert "workbook" not in outputs
    assert not (tmp_path / "manual_feature_selection.xlsx").exists()


def test_s05_dispatches_csv_selection_and_freezes_json(tmp_path):
    import s04_feature_selection as s04
    import s05_train_final_model as s05

    outputs = s04.export_full_feature_ranking(tmp_path, _ranking_payload()["ranking"])
    selected = {"COMM_AMB_AC", "GREEN_CORR"}
    _select(outputs["selection_csv"], selected)
    frame = pd.DataFrame({name: [0.0, 1.0] for name in catalog.model_candidate_names()})
    frame["target"] = [0, 1]
    frame["sample_name"] = ["a", "b"]
    frame["feature_pool_version"] = catalog.FEATURE_POOL_VERSION

    actual, provenance = s05.load_manual_feature_selection(
        outputs["selection_csv"], outputs["json"], frame, frame
    )

    assert actual == [name for name in catalog.model_candidate_names() if name in selected]
    frozen = json.loads((tmp_path / "manual_selected_features.json").read_text(encoding="utf-8"))
    assert frozen["selected_features"] == actual
    assert provenance["selection_source_type"] == "csv"


def test_s05_rejects_non_csv_manual_selection(tmp_path):
    import s05_train_final_model as s05

    ranking_path = tmp_path / "feature_ranking_full.json"
    _write_ranking(ranking_path)
    xlsx_path = tmp_path / "manual_feature_selection.xlsx"
    xlsx_path.write_bytes(b"legacy")
    frame = pd.DataFrame({name: [0.0] for name in catalog.model_candidate_names()})
    frame["feature_pool_version"] = catalog.FEATURE_POOL_VERSION

    with pytest.raises(ValueError, match="CSV-only"):
        s05.load_manual_feature_selection(xlsx_path, ranking_path, frame, frame)
