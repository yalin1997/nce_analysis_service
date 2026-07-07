from pathlib import Path

ANALYSIS_CORE_DIR = Path(__file__).resolve().parents[2] / "analysis_core"


def test_analysis_core_never_references_nce_analysis():
    py_files = list(ANALYSIS_CORE_DIR.rglob("*.py"))
    assert py_files, "analysis_core package not found"
    for source_file in py_files:
        content = source_file.read_text(encoding="utf-8")
        assert "nce_analysis" not in content, (
            f"{source_file} references nce_analysis; analysis_core must stay domain-free"
        )
