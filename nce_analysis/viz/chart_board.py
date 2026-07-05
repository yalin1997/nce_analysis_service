import html as html_lib

import pandas as pd
import plotly.graph_objects as go

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.ratio_threshold import LITHO_POINT_KEY_COLUMNS
from nce_analysis.root_cause.base import resolve_detail_history
from nce_analysis.schema import AnalysisResult, RootCauseDetail

_CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
_OTHER_COLOR = "#898781"
_OTHER_LABEL = "Other"
_CRITICAL_COLOR = "#d03b3b"
_SURFACE_RING_COLOR = "#fcfcfb"

_SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

_LITHO_SELF_TYPES = ("LITHO_CHUCK_ISSUE", "LITHO_CHUCK_CONTAMINATION", "LITHO_TOOL_ISSUE")

_CSS = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #f9f9f7; color: #0b0b0b; margin: 0; }
.header { padding: 16px 24px; border-bottom: 1px solid #e1e0d9; }
.layout { display: flex; }
.sidebar { width: 280px; border-right: 1px solid #e1e0d9; padding: 12px; }
.sidebar-row { padding: 8px; border-radius: 4px; cursor: pointer; margin-bottom: 4px; }
.sidebar-row:hover { background: #efefef; }
.sidebar-row.selected { background: #e1e0d9; }
.sidebar-footer { color: #898781; font-size: 0.8em; margin-top: 12px; }
.main { flex: 1; padding: 16px 24px; }
.hidden { display: none; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; margin-left: 4px; }
.badge-type { background: #e1e0d9; color: #52514e; }
.badge-warning { background: #fab219; color: #0b0b0b; }
.empty-note { color: #898781; font-style: italic; }
.chart-caption { color: #52514e; font-size: 0.85em; }
table { border-collapse: collapse; }
td, th { padding: 4px 8px; border-bottom: 1px solid #e1e0d9; text-align: left; }
.toggle-table-btn { margin-top: 8px; }
"""

_JS = """
function showDetailPanel(rank) {
  document.querySelectorAll('.detail-panel').forEach(function(el) {
    el.classList.add('hidden');
  });
  document.querySelectorAll('.sidebar-row').forEach(function(el) {
    el.classList.remove('selected');
  });
  var panel = document.getElementById('detail-panel-' + rank);
  if (panel) { panel.classList.remove('hidden'); }
  var row = document.querySelector('.sidebar-row[data-rank="' + rank + '"]');
  if (row) { row.classList.add('selected'); }
}
"""


def _sequential_colorscale() -> list[list[float | str]]:
    n = len(_SEQUENTIAL_BLUE_STEPS)
    return [[i / (n - 1), color] for i, color in enumerate(_SEQUENTIAL_BLUE_STEPS)]


def _suspect_label(detail: RootCauseDetail) -> str:
    if detail.Suspect_Pre_ChamberID == "N/A":
        return detail.Suspect_Pre_ToolID
    if detail.Root_Cause_Type in _LITHO_SELF_TYPES:
        return f"{detail.Suspect_Pre_ToolID} / Chuck {detail.Suspect_Pre_ChamberID}"
    return f"{detail.Suspect_Pre_ToolID} / {detail.Suspect_Pre_ChamberID}"


class _PlotlyEmbedder:
    def __init__(self) -> None:
        self._embedded = False

    def to_html(self, fig: go.Figure) -> str:
        include = "inline" if not self._embedded else False
        self._embedded = True
        return fig.to_html(
            full_html=False, include_plotlyjs=include, config={"displayModeBar": False}
        )


def _assign_global_colors(
    summary: list[RootCauseDetail], histories: list[pd.DataFrame]
) -> dict[str, str]:
    suspect_labels: list[str] = []
    wafer_counts: dict[str, int] = {}
    first_seen_order: list[str] = []

    for resolved in histories:
        if resolved is None or resolved.empty:
            continue
        for label, group in resolved.groupby("group_label"):
            if label not in wafer_counts:
                wafer_counts[label] = 0
                first_seen_order.append(label)
            wafer_counts[label] += len(group)
            if group["is_suspect_group"].any() and label not in suspect_labels:
                suspect_labels.append(label)

    other_labels = [label for label in first_seen_order if label not in suspect_labels]
    other_labels.sort(key=lambda label: (-wafer_counts[label], first_seen_order.index(label)))

    ordered_labels = suspect_labels + other_labels
    n_slots = len(_CATEGORICAL_COLORS)
    return {label: _CATEGORICAL_COLORS[i] for i, label in enumerate(ordered_labels[:n_slots])}


def _add_marker_trace(fig: go.Figure, sub_df: pd.DataFrame, color: str, legend_name: str) -> None:
    symbols = ["diamond" if a else "circle" for a in sub_df["is_anomaly"]]
    status = ["above spec limit" if a else "within spec" for a in sub_df["is_anomaly"]]
    customdata = list(zip(sub_df["WaferID"], sub_df["group_label"], status, sub_df["time"]))
    fig.add_trace(
        go.Scatter(
            x=[sub_df["group_label"], sub_df["time"]],
            y=sub_df["NCE_Value"],
            mode="markers",
            name=legend_name,
            marker=dict(
                color=color,
                size=10,
                symbol=symbols,
                line=dict(width=2, color=_SURFACE_RING_COLOR),
            ),
            customdata=customdata,
            hovertemplate=(
                "WaferID: %{customdata[0]}<br>"
                "NCE_Value: %{y}<br>"
                "Time: %{customdata[3]}<br>"
                "Group: %{customdata[1]}<br>"
                "Status: %{customdata[2]}<extra></extra>"
            ),
        )
    )


def _spc_chart_fragment(
    resolved: pd.DataFrame,
    color_map: dict[str, str],
    config: AnalysisConfig,
    embedder: _PlotlyEmbedder,
) -> str:
    if resolved is None or resolved.empty:
        return '<p class="empty-note">No history data available for this suspect.</p>'

    suspect_label = None
    if resolved["is_suspect_group"].any():
        suspect_label = resolved.loc[resolved["is_suspect_group"], "group_label"].iloc[0]

    distinct_labels = resolved["group_label"].unique().tolist()
    fig = go.Figure()

    other_rows = []
    for label in distinct_labels:
        sub_df = resolved[resolved["group_label"] == label]
        if label in color_map:
            legend_name = label
            if suspect_label is not None and label == suspect_label:
                legend_name = f"{label} (suspect)"
            _add_marker_trace(fig, sub_df, color_map[label], legend_name)
        else:
            other_rows.append(sub_df)

    if other_rows:
        _add_marker_trace(fig, pd.concat(other_rows), _OTHER_COLOR, _OTHER_LABEL)

    fig.add_hline(
        y=config.spec_threshold,
        line_dash="dash",
        line_color=_CRITICAL_COLOR,
        annotation_text=f"Spec limit: {config.spec_threshold}",
        annotation_position="top left",
    )

    fig.update_layout(
        showlegend=len(distinct_labels) > 1,
        margin=dict(l=40, r=20, t=40, b=40),
        height=420,
        plot_bgcolor=_SURFACE_RING_COLOR,
        paper_bgcolor=_SURFACE_RING_COLOR,
    )
    fig.update_xaxes(title_text="Group / Time")
    fig.update_yaxes(title_text="NCE_Value")

    chart_html = embedder.to_html(fig)
    caption = '<p class="chart-caption">○ within spec &middot; ◆ above spec limit</p>'
    return f'<div class="spc-chart">{chart_html}</div>{caption}'


def _heatmap_fragment(
    result: AnalysisResult,
    long_df: pd.DataFrame,
    config: AnalysisConfig,
    embedder: _PlotlyEmbedder,
) -> str:
    litho_points = long_df.drop_duplicates(subset=LITHO_POINT_KEY_COLUMNS).copy()
    litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

    grouped = (
        litho_points.groupby(["X_Posi", "Y_Posi"])
        .agg(
            anomalous_wafer_count=("is_anomaly", "sum"),
            total_wafer_count=("is_anomaly", "count"),
        )
        .reset_index()
    )
    grouped["anomaly_ratio"] = grouped["anomalous_wafer_count"] / grouped["total_wafer_count"]

    flagged_coords = {(h.X_Posi, h.Y_Posi) for h in result.Insufficient_Sample_Hotspots}
    for detail in result.Details:
        flagged_coords.update(tuple(c) for c in detail.Affected_Coordinates)

    grouped["flagged"] = pd.Series(
        list(zip(grouped["X_Posi"], grouped["Y_Posi"])), index=grouped.index
    ).isin(flagged_coords)
    symbols = ["square" if f else "circle" for f in grouped["flagged"]]

    fig = go.Figure(
        data=go.Scatter(
            x=grouped["X_Posi"],
            y=grouped["Y_Posi"],
            mode="markers",
            marker=dict(
                color=grouped["anomaly_ratio"],
                colorscale=_sequential_colorscale(),
                size=14,
                symbol=symbols,
                showscale=True,
                colorbar=dict(title="Anomaly ratio"),
                line=dict(width=1, color=_SURFACE_RING_COLOR),
            ),
            customdata=list(
                zip(
                    grouped["anomaly_ratio"],
                    grouped["anomalous_wafer_count"],
                    grouped["total_wafer_count"],
                )
            ),
            hovertemplate=(
                "X: %{x}<br>Y: %{y}<br>"
                "Anomaly ratio: %{customdata[0]:.2f}<br>"
                "Anomalous: %{customdata[1]}/%{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        height=500,
        plot_bgcolor=_SURFACE_RING_COLOR,
        paper_bgcolor=_SURFACE_RING_COLOR,
    )
    fig.update_xaxes(title_text="X_Posi")
    fig.update_yaxes(title_text="Y_Posi")

    chart_html = embedder.to_html(fig)

    flagged_rows = grouped[grouped["flagged"]]
    if flagged_rows.empty:
        table_html = '<p class="empty-note">No flagged coordinates.</p>'
    else:
        rows_html = "".join(
            f"<tr><td>{r.X_Posi}</td><td>{r.Y_Posi}</td>"
            f"<td>{r.anomaly_ratio:.3f}</td><td>{r.anomalous_wafer_count}</td>"
            f"<td>{r.total_wafer_count}</td></tr>"
            for r in flagged_rows.itertuples()
        )
        table_html = (
            '<table class="flagged-table">'
            "<thead><tr><th>X_Posi</th><th>Y_Posi</th><th>Anomaly Ratio</th>"
            "<th>Anomalous</th><th>Total</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )

    return (
        '<div class="heatmap-section">'
        f'<div class="heatmap-chart">{chart_html}</div>'
        '<button class="toggle-table-btn" '
        "onclick=\"document.getElementById('flagged-table').classList.toggle('hidden')\">"
        "View as table</button>"
        f'<div id="flagged-table" class="hidden">{table_html}</div>'
        "</div>"
    )


def _metrics_table_html(metrics: dict[str, float]) -> str:
    if not metrics:
        return '<p class="empty-note">No metrics.</p>'
    rows = "".join(
        f"<tr><td>{html_lib.escape(str(k))}</td><td>{v}</td></tr>" for k, v in metrics.items()
    )
    return f'<table class="metrics-table"><tbody>{rows}</tbody></table>'


def _coordinates_html(coords: list[tuple[float, float]]) -> str:
    if not coords:
        return '<p class="empty-note">No affected coordinates.</p>'
    items = "".join(f"<li>({x}, {y})</li>" for x, y in coords)
    return f'<ul class="coord-list">{items}</ul>'


def _detail_panel_html(rank: int, detail: RootCauseDetail, chart_fragment: str) -> str:
    review_badge = (
        '<span class="badge badge-warning">Requires Manual Review</span>'
        if detail.Requires_Manual_Review
        else ""
    )
    hidden = "" if rank == 1 else " hidden"
    return (
        f'<div class="detail-panel{hidden}" id="detail-panel-{rank}" data-rank="{rank}">'
        f"<h2>{html_lib.escape(_suspect_label(detail))}</h2>"
        f'<span class="badge badge-type">{detail.Root_Cause_Type}</span>'
        f"<p>Confidence Score: {detail.Confidence_Score:.1f}</p>"
        f"{review_badge}"
        "<h3>Metrics</h3>"
        f"{_metrics_table_html(detail.Metrics)}"
        "<h3>Affected Coordinates</h3>"
        f"{_coordinates_html(detail.Affected_Coordinates)}"
        "<h3>SPC Chart</h3>"
        f"{chart_fragment}"
        "</div>"
    )


def _sidebar_row_html(rank: int, detail: RootCauseDetail) -> str:
    review_badge = (
        '<span class="badge badge-warning">!</span>' if detail.Requires_Manual_Review else ""
    )
    return (
        f'<div class="sidebar-row" data-rank="{rank}" onclick="showDetailPanel({rank})">'
        f'<span class="rank">#{rank}</span> '
        f'<span class="suspect-label">{html_lib.escape(_suspect_label(detail))}</span> '
        f'<span class="badge badge-type">{detail.Root_Cause_Type}</span> '
        f'<span class="confidence">{detail.Confidence_Score:.1f}</span>'
        f"{review_badge}"
        "</div>"
    )


def render_chart_board(
    result: AnalysisResult, long_df: pd.DataFrame, config: AnalysisConfig
) -> str:
    """Render a self-contained HTML report (sidebar of ranked suspects, a
    per-suspect SPC chart, and a wafer-coordinate anomaly-ratio heatmap).
    Returns the HTML document as a string; the caller decides whether to
    write it to disk. long_df is the preprocessed (WideHistoryReshape)
    output — same shape pipeline.run consumes internally."""
    embedder = _PlotlyEmbedder()

    histories = [resolve_detail_history(long_df, detail, config) for detail in result.Summary]
    color_map = _assign_global_colors(result.Summary, histories)

    sidebar_rows = []
    detail_panels = []
    for i, (detail, resolved) in enumerate(zip(result.Summary, histories), start=1):
        sidebar_rows.append(_sidebar_row_html(i, detail))
        chart_fragment = _spc_chart_fragment(resolved, color_map, config, embedder)
        detail_panels.append(_detail_panel_html(i, detail, chart_fragment))

    if not result.Summary:
        sidebar_rows.append('<p class="empty-note">No root-cause suspects identified.</p>')
        detail_panels.append('<p class="empty-note">No details to display.</p>')

    heatmap_fragment = _heatmap_fragment(result, long_df, config, embedder)

    footer = ""
    if len(result.Details) > len(result.Summary):
        footer = (
            f'<p class="sidebar-footer">Showing top {len(result.Summary)} '
            f"of {len(result.Details)}</p>"
        )

    config_summary = (
        f"spec_threshold={config.spec_threshold} &middot; "
        f"root_cause_granularity={config.root_cause_granularity} &middot; "
        f"root_cause_strategy={config.root_cause_strategy} &middot; "
        f"drift_strategy={config.drift_strategy}"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NCE Root-Cause Chart Board</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
<h1>NCE Root-Cause Chart Board</h1>
<p>Generated: {result.Generated_At.isoformat()}</p>
<p>{config_summary}</p>
</div>
<div class="layout">
<div class="sidebar">
{''.join(sidebar_rows)}
{footer}
</div>
<div class="main">
{''.join(detail_panels)}
<h2>Wafer Measurement Map</h2>
{heatmap_fragment}
</div>
</div>
<script>{_JS}</script>
</body>
</html>
"""
