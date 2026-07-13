from pathlib import Path

import numpy as np
import pandas as pd


def test_s01_exports_split_analysis_png(tmp_path):
    import s01_data_split as s01

    split = {
        "train": [
            {"sample_name": "a", "target": 0, "duration_sec": 20.0},
            {"sample_name": "b", "target": 1, "duration_sec": 22.0},
        ],
        "valid": [{"sample_name": "c", "target": 0, "duration_sec": 18.0}],
        "test": [{"sample_name": "d", "target": 1, "duration_sec": 24.0}],
    }

    outputs = s01.export_split_analysis_plot(split, tmp_path)

    assert Path(outputs["png"]).exists()
    assert Path(outputs["png"]).suffix == ".png"
    assert Path(outputs["source_data"]).exists()
    assert Path(outputs["manifest"]).exists()
    assert Path(outputs["qa"]).exists()


def test_s03_exports_feature_pool_analysis_png(tmp_path):
    import s03_extract_feature_pool as s03
    from stage2_feature_catalog import model_candidate_names

    names = model_candidate_names()
    rng = np.random.default_rng(42)
    frames = {}
    for split_name, n_rows in [("train", 30), ("valid", 16), ("test", 12)]:
        target = np.arange(n_rows) % 2
        data = {name: rng.normal(target * 0.2, 1.0, n_rows) for name in names}
        data.update({
            "sample_name": [f"{split_name}_{i // 2}" for i in range(n_rows)],
            "target": target,
        })
        frames[split_name] = pd.DataFrame(data)

    outputs = s03.export_feature_pool_analysis_plot(frames, tmp_path)

    assert Path(outputs["png"]).exists()
    assert Path(outputs["png"]).suffix == ".png"
    assert Path(outputs["source_data"]).exists()
    assert Path(outputs["manifest"]).exists()
    assert Path(outputs["qa"]).exists()
