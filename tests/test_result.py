from nce_analysis.config import AnalysisConfig
from nce_analysis.result import aggregate_results
from nce_analysis.schema import InsufficientSampleHotspot, RootCauseDetail


def _detail(score, coord=(0.0, 0.0), chamber="ChamberA"):
    return RootCauseDetail(
        Suspect_Pre_ToolID="CMP_01",
        Suspect_Pre_ChamberID=chamber,
        Suspect_Pre_StepID="3580.01",
        Confidence_Score=score,
        Root_Cause_Type="SPECIFIC_CHAMBER_DEFECT",
        Affected_Coordinates=[coord],
        Metrics={},
    )


def test_aggregate_results_sorts_details_by_confidence_descending():
    details = [_detail(50.0), _detail(90.0), _detail(70.0)]
    config = AnalysisConfig(summary_top_n=2)

    result = aggregate_results(details, config)

    assert [d.Confidence_Score for d in result.Details] == [90.0, 70.0, 50.0]
    assert result.Config_Used.summary_top_n == 2


def test_aggregate_results_merges_same_suspect_into_one_summary_entry():
    details = [
        _detail(90.0, coord=(0.0, 0.0)),
        _detail(70.0, coord=(6.0, 0.0)),
        _detail(60.0, coord=(12.0, 0.0), chamber="ChamberB"),
    ]
    config = AnalysisConfig(summary_top_n=5)

    result = aggregate_results(details, config)

    assert len(result.Summary) == 2
    top = result.Summary[0]
    assert top.Suspect_Pre_ChamberID == "ChamberA"
    assert top.Confidence_Score == 90.0
    assert set(top.Affected_Coordinates) == {(0.0, 0.0), (6.0, 0.0)}
    assert result.Summary[1].Suspect_Pre_ChamberID == "ChamberB"
    assert len(result.Details) == 3


def test_aggregate_results_handles_empty_details_and_carries_insufficient():
    hotspot = InsufficientSampleHotspot(
        X_Posi=0.0,
        Y_Posi=0.0,
        anomaly_ratio=1.0,
        anomalous_wafer_count=2,
        total_wafer_count=2,
    )

    result = aggregate_results([], AnalysisConfig(), [hotspot])

    assert result.Summary == []
    assert result.Details == []
    assert result.Insufficient_Sample_Hotspots == [hotspot]
