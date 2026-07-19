def test_model_fingerprint_hashes_complete_feature_pool_file(tmp_path):
    import s05_train_final_model as s05

    splits_path = tmp_path / "splits.json"
    feature_path = tmp_path / "feature_pool_train.csv"
    splits_path.write_text('{"train":[]}', encoding="utf-8")
    feature_path.write_bytes(b"a" * (4 * 1024 * 1024) + b"x")

    before = s05.build_fingerprint(tmp_path, feature_path, splits_path)
    feature_path.write_bytes(b"a" * (4 * 1024 * 1024) + b"y")
    after = s05.build_fingerprint(tmp_path, feature_path, splits_path)

    assert before["feature_pool_train_sha256"] != after["feature_pool_train_sha256"]
    assert before["splits_sha256"] == after["splits_sha256"]


def test_active_optimization_plan_documents_manual_resume_command():
    from pathlib import Path
    from stage2_feature_catalog import FEATURE_POOL_VERSION

    plan = Path("SINGLE_WINDOW_98_FEATURE_OPTIMIZATION_PLAN.md").read_text(
        encoding="utf-8"
    )

    assert FEATURE_POOL_VERSION in plan
    assert "stage2_interpretable_v8" not in plan
    assert "特征池为 v8/126" not in plan
    assert "保存后重新运行同一命令" not in plan
    assert "--feature_selection_mode manual" in plan
    assert "--manual_feature_file artifacts/manual_feature_selection.csv" in plan
    assert "--skip s01,s03,s04" in plan
