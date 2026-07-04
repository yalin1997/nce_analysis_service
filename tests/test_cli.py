import json

import pandas as pd

from nce_analysis.cli import main


def _write_minimal_csv(path):
    row = {
        "PartID": "P1",
        "WaferID": "W1",
        "StageID": "LITHO_M1",
        "StepID": "S1",
        "ToolID": "LITHO_A",
        "ChuckID": "CHK_1",
        "Execute_Time": "2026-01-01",
        "X_Posi": 0.0,
        "Y_Posi": 0.0,
        "NCE_Value": 5.0,
        "Pre_StageID_1": "CMP",
        "Pre_StepID_1": "3580.01",
        "Pre_ToolID_1": "CMP_01",
        "Pre_ChamberID_1": "ChamberA",
        "Pre_Execute_Time_1": "2025-12-01",
    }
    pd.DataFrame([row]).to_csv(path, index=False)


def test_main_writes_result_json_to_output_file(tmp_path):
    csv_path = tmp_path / "data.csv"
    out_path = tmp_path / "result.json"
    _write_minimal_csv(csv_path)

    exit_code = main(["--input", str(csv_path), "--output", str(out_path)])

    assert exit_code == 0
    result = json.loads(out_path.read_text())
    assert "Summary" in result
    assert "Details" in result


def test_main_prints_result_json_to_stdout_when_no_output(tmp_path, capsys):
    csv_path = tmp_path / "data.csv"
    _write_minimal_csv(csv_path)

    exit_code = main(["--input", str(csv_path)])

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert "Summary" in result


def test_main_applies_config_yaml_override(tmp_path):
    csv_path = tmp_path / "data.csv"
    out_path = tmp_path / "result.json"
    config_path = tmp_path / "config.yaml"
    _write_minimal_csv(csv_path)
    config_path.write_text("summary_top_n: 3\n")

    exit_code = main(
        ["--input", str(csv_path), "--config", str(config_path), "--output", str(out_path)]
    )

    assert exit_code == 0
    result = json.loads(out_path.read_text())
    assert result["Config_Used"]["summary_top_n"] == 3


def test_main_reports_error_for_unsupported_input_extension(tmp_path, capsys):
    bad_path = tmp_path / "data.txt"
    bad_path.write_text("not real data")

    exit_code = main(["--input", str(bad_path)])

    assert exit_code == 1
    assert "Error loading input file" in capsys.readouterr().err


def test_main_reports_error_for_invalid_config_field(tmp_path, capsys):
    csv_path = tmp_path / "data.csv"
    config_path = tmp_path / "config.yaml"
    _write_minimal_csv(csv_path)
    config_path.write_text("typo_field: 1\n")

    exit_code = main(["--input", str(csv_path), "--config", str(config_path)])

    assert exit_code == 1
    assert "Error loading config file" in capsys.readouterr().err


def test_main_reports_error_for_preprocessing_failure(tmp_path, capsys):
    csv_path = tmp_path / "data.csv"
    # No Pre_*_i history columns at all -> discover_history_levels finds
    # none -> PreprocessingError inside pipeline.run.
    pd.DataFrame(
        [
            {
                "PartID": "P1",
                "WaferID": "W1",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "Execute_Time": "2026-01-01",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
            }
        ]
    ).to_csv(csv_path, index=False)

    exit_code = main(["--input", str(csv_path)])

    assert exit_code == 1
    assert "Error running analysis pipeline" in capsys.readouterr().err
