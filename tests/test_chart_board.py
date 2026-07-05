from datetime import datetime

import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.schema import AnalysisResult, InsufficientSampleHotspot, RootCauseDetail
from nce_analysis.viz.chart_board import render_chart_board


def _detail(**overrides):
    defaults = dict(
        Suspect_Pre_ToolID="CMP_01",
        Suspect_Pre_ChamberID="ChamberA",
        Suspect_Pre_StepID="STEP1",
        Confidence_Score=90.0,
        Root_Cause_Type="SPECIFIC_CHAMBER_DEFECT",
        Affected_Coordinates=[(0.0, 0.0)],
        Metrics={"p_value": 0.01},
    )
    defaults.update(overrides)
    return RootCauseDetail(**defaults)


def _long_df_rows(n_peers=1, step_id="STEP1", coord=(0.0, 0.0)):
    rows = [
        {
            "WaferID": "W_suspect",
            "X_Posi": coord[0],
            "Y_Posi": coord[1],
            "NCE_Value": 20.0,
            "Pre_ToolID": "CMP_01",
            "Pre_ChamberID": "ChamberA",
            "Pre_StepID": step_id,
            "Pre_Execute_Time": "2026-01-01",
        }
    ]
    for i in range(n_peers):
        rows.append(
            {
                "WaferID": f"W_peer_{i}",
                "X_Posi": coord[0],
                "Y_Posi": coord[1],
                "NCE_Value": 5.0,
                "Pre_ToolID": f"CMP_{i + 2:02d}",
                "Pre_ChamberID": f"Chamber{i}",
                "Pre_StepID": step_id,
                "Pre_Execute_Time": "2026-01-01",
            }
        )
    return pd.DataFrame(rows)


def _result(details=None, summary=None, insufficient=None):
    details = details if details is not None else [_detail()]
    summary = summary if summary is not None else details
    return AnalysisResult(
        Summary=summary,
        Details=details,
        Insufficient_Sample_Hotspots=insufficient or [],
        Generated_At=datetime(2026, 7, 5, 12, 0, 0),
        Config_Used=AnalysisConfig(),
    )


def test_render_chart_board_returns_full_html_document():
    html = render_chart_board(_result(), _long_df_rows(), AnalysisConfig())

    assert "<html" in html.lower()


def test_render_chart_board_embeds_plotly_js_exactly_once():
    html = render_chart_board(_result(), _long_df_rows(), AnalysisConfig())

    assert html.count("plotly.js v") == 1


def test_render_chart_board_one_sidebar_row_per_summary_entry():
    detail1 = _detail(Suspect_Pre_StepID="STEP1", Affected_Coordinates=[(0.0, 0.0)])
    detail2 = _detail(Suspect_Pre_StepID="STEP1", Affected_Coordinates=[(1.0, 1.0)])
    result = _result(details=[detail1, detail2], summary=[detail1])

    long_df = pd.concat([_long_df_rows(coord=(0.0, 0.0)), _long_df_rows(coord=(1.0, 1.0))])
    html = render_chart_board(result, long_df, AnalysisConfig())

    assert html.count('class="sidebar-row"') == 1


def test_render_chart_board_includes_spec_threshold_reference_line_value():
    config = AnalysisConfig(spec_threshold=17.5)
    html = render_chart_board(_result(), _long_df_rows(), config)

    assert "17.5" in html


def test_render_chart_board_suspect_always_gets_dedicated_slot_and_suffix():
    long_df = _long_df_rows(n_peers=9)
    result = _result()

    html = render_chart_board(result, long_df, AnalysisConfig())

    assert "CMP_01|ChamberA (suspect)" in html
    assert '"Other"' in html


def test_render_chart_board_omits_legend_for_single_group_panel():
    long_df = _long_df_rows(n_peers=0)
    result = _result()

    html = render_chart_board(result, long_df, AnalysisConfig())

    assert '"showlegend": false' in html or '"showlegend":false' in html


def test_render_chart_board_empty_summary_does_not_crash():
    result = _result(details=[], summary=[])

    html = render_chart_board(result, _long_df_rows(), AnalysisConfig())

    assert "<html" in html.lower()


def test_render_chart_board_shows_requires_manual_review_badge():
    detail = _detail(Requires_Manual_Review=True)
    result = _result(details=[detail], summary=[detail])

    html = render_chart_board(result, _long_df_rows(), AnalysisConfig())

    assert "Requires Manual Review" in html or "Manual Review" in html


def test_render_chart_board_footer_shows_top_n_of_total_when_details_exceed_summary():
    detail1 = _detail(Affected_Coordinates=[(0.0, 0.0)])
    detail2 = _detail(Suspect_Pre_ToolID="CMP_02", Affected_Coordinates=[(1.0, 1.0)])
    result = _result(details=[detail1, detail2], summary=[detail1])

    long_df = pd.concat([_long_df_rows(coord=(0.0, 0.0)), _long_df_rows(coord=(1.0, 1.0))])
    html = render_chart_board(result, long_df, AnalysisConfig())

    assert "Showing top 1 of 2" in html


def test_render_chart_board_flagged_coordinate_appears_in_table_toggle():
    insufficient = [
        InsufficientSampleHotspot(
            X_Posi=9.0,
            Y_Posi=9.0,
            anomaly_ratio=0.2,
            anomalous_wafer_count=1,
            total_wafer_count=5,
        )
    ]
    result = _result(insufficient=insufficient)
    long_df = pd.concat(
        [
            _long_df_rows(),
            pd.DataFrame(
                [
                    {
                        "WaferID": "W_flagged",
                        "X_Posi": 9.0,
                        "Y_Posi": 9.0,
                        "NCE_Value": 5.0,
                        "Pre_ToolID": "CMP_09",
                        "Pre_ChamberID": "ChamberZ",
                        "Pre_StepID": "STEP1",
                        "Pre_Execute_Time": "2026-01-01",
                    }
                ]
            ),
        ]
    )

    html = render_chart_board(result, long_df, AnalysisConfig())

    assert "9.0" in html


def test_render_chart_board_resolve_history_empty_renders_text_note_not_chart():
    detail = _detail(Suspect_Pre_StepID="NONEXISTENT_STEP")
    result = _result(details=[detail], summary=[detail])

    html = render_chart_board(result, _long_df_rows(), AnalysisConfig())

    assert "No history data available" in html
