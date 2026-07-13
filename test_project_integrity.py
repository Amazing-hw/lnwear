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
