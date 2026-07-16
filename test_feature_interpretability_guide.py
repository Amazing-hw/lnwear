import re
from pathlib import Path

import stage2_feature_catalog as catalog


ROOT = Path(__file__).resolve().parent
GUIDE_PATH = ROOT / "FEATURE_INTERPRETABILITY_GUIDE.md"
FIELD_LABELS = [
    "定义/公式",
    "生理与物理意义",
    "佩戴判别预期",
    "鲁棒性价值",
    "混淆与泛化风险",
    "工程与人工筛选建议",
]


def _feature_sections(text):
    matches = list(re.finditer(r"^### `([^`]+)`\s*$", text, flags=re.MULTILINE))
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1), text[match.end():end]))
    return sections


def test_feature_interpretability_guide_covers_catalog_in_exact_order():
    text = GUIDE_PATH.read_text(encoding="utf-8")
    documented = [name for name, _section in _feature_sections(text)]

    assert documented == catalog.model_candidate_names()
    assert len(documented) == 126
    assert len(set(documented)) == len(documented)


def test_every_feature_section_contains_all_interpretability_fields():
    text = GUIDE_PATH.read_text(encoding="utf-8")

    for name, section in _feature_sections(text):
        for label in FIELD_LABELS:
            assert f"- **{label}**：" in section, f"{name} missing {label}"


def test_feature_guide_documents_version_and_manual_selection_boundary():
    text = GUIDE_PATH.read_text(encoding="utf-8")

    assert catalog.FEATURE_POOL_VERSION in text
    assert "manual_feature_selection.csv" in text
    assert "统计证据" in text
    assert "不是单一生理因果" in text


def test_readme_links_feature_interpretability_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "FEATURE_INTERPRETABILITY_GUIDE.md" in readme
