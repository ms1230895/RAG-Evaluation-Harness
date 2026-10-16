"""Plotly figures for the dashboard (Step 15). No streamlit import.

Each function takes stored summary dicts (the output of metrics.rate_summary /
mean_summary, as returned by evalharness.dashboard_data) and only DRAWS them.
No metric is computed here.

Honesty rules built into every rate figure:
- the y-axis always runs from 0 to 100%, never a truncated range;
- every bar / point has its 95% Wilson interval as an error bar;
- k/n is in the hover text and under the bar;
- a rate with n = 0 is drawn as "n = 0", not as 0%.

The colours are presentation constants (Okabe-Ito, colour-blind safe), not
experiment values: one colour per metric and one per query type, used on every page.
"""

import plotly.graph_objects as go

# Okabe-Ito palette
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#7F7F7F")

# The only metric names the trend chart accepts (an allow-list, never free text).
# group: which dict of a trend row holds the summary ("rates" have a Wilson interval,
# "means" have none).
TREND_METRICS = {
    "ungrounded": {"label": "Ungrounded rate", "group": "rates", "color": VERMILLION},
    "bad_answer": {"label": "Bad answer rate (ungrounded or irrelevant)", "group": "rates",
                   "color": PURPLE},
    "irrelevant": {"label": "Irrelevant rate", "group": "rates", "color": ORANGE},
    "in_context": {"label": "Source in prompt (answerable)", "group": "rates", "color": SKY},
    "hit@4": {"label": "Hit rate @4 (answerable)", "group": "rates", "color": BLUE},
    "faithfulness": {"label": "Mean faithfulness", "group": "means", "color": GREEN},
    "relevance": {"label": "Mean relevance", "group": "means", "color": ORANGE},
    "mrr": {"label": "MRR (answerable)", "group": "means", "color": BLUE},
}

# One colour per query type, the same on every page.
TYPE_COLORS = {"lookup": BLUE, "multi_chunk": GREEN, "paraphrased": ORANGE,
               "unanswerable": PURPLE, "all": GREY, "answerable": SKY}
CAUSE_COLORS = {"retrieval_miss": VERMILLION, "generation_error": BLUE}
TEMPLATE = "plotly_white"


def _kn(summary: dict) -> str:
    """'7/80', or 'n = 0' when nothing was judged."""
    return "n = 0" if summary["n"] == 0 else f"{summary['successes']}/{summary['n']}"


def _error_arrays(summaries: list[dict]) -> tuple[list[float], list[float]]:
    """(upper, lower) error-bar lengths from the stored Wilson bounds; 0 where n = 0."""
    upper = [s["ci_high"] - s["rate"] if s["rate"] is not None else 0 for s in summaries]
    lower = [s["rate"] - s["ci_low"] if s["rate"] is not None else 0 for s in summaries]
    return upper, lower


def _rate_axis(fig: go.Figure) -> None:
    """Full 0-100% axis: a rate chart never starts above zero."""
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="rate")


def intervals_overlap(rates: list[dict]) -> bool:
    """True when EVERY pair of stored Wilson intervals overlaps (so no pair is separated).

    Rates with n = 0 are ignored; fewer than two usable rates gives False.
    """
    spans = [(r["ci_low"], r["ci_high"]) for r in rates if r.get("rate") is not None]
    if len(spans) < 2:
        return False
    return all(a_low <= b_high and b_low <= a_high
               for i, (a_low, a_high) in enumerate(spans) for b_low, b_high in spans[i + 1:])


def make_trend_figure(rows: list[dict], metric: str, confidence: float) -> go.Figure:
    """One point per run (x = start time), one trace per config name.

    `rows` come from dashboard_data.trend_table (all one split). A mean metric has no
    interval and says so in the title.
    """
    if metric not in TREND_METRICS:
        raise ValueError(f"unknown trend metric {metric!r}; allowed: {sorted(TREND_METRICS)}")
    spec = TREND_METRICS[metric]
    is_rate = spec["group"] == "rates"
    pct = round(confidence * 100)

    fig = go.Figure()
    omitted = 0
    for config_name in sorted({r["config_name"] for r in rows}):
        points = []
        for row in rows:
            if row["config_name"] != config_name:
                continue
            summary = row[spec["group"]][metric]
            value = summary["rate"] if is_rate else summary["mean"]
            if value is None:
                omitted += 1   # n = 0: not drawn as 0
                continue
            points.append((row, summary, value))
        if not points:
            continue
        error_y = None
        if is_rate:
            upper, lower = _error_arrays([p[1] for p in points])
            error_y = {"type": "data", "array": upper, "arrayminus": lower, "visible": True}
        fig.add_trace(go.Scatter(
            x=[p[0]["timestamp"] for p in points], y=[p[2] for p in points], mode="markers",
            name=config_name, error_y=error_y,
            customdata=[[p[0]["run_id"], p[0]["git_commit"], p[0]["judge_version"] or "-",
                         _kn(p[1]) if is_rate else f"n = {p[1]['n']}"] for p in points],
            hovertemplate=("%{customdata[0]}<br>config: " + config_name
                           + "<br>git %{customdata[1]}<br>judge version: %{customdata[2]}"
                           + "<br>value: %{y:.3f} (%{customdata[3]})<extra></extra>"),
        ))
    suffix = f"{pct}% Wilson interval" if is_rate else "mean, no interval"
    fig.update_layout(template=TEMPLATE, title=f"{spec['label']} per run, {suffix}",
                      xaxis_title="run start (UTC)", legend_title="config",
                      margin={"t": 60})
    if is_rate:
        _rate_axis(fig)
    else:
        fig.update_yaxes(range=[0, 1], title_text="mean (0 to 1)")
    if omitted:
        fig.add_annotation(text=f"{omitted} run(s) with n = 0 not drawn", xref="paper",
                           yref="paper", x=0, y=1.08, showarrow=False)
    return fig


def make_breakdown_figure(report: dict, group_names: list[str]) -> go.Figure:
    """Ungrounded rate per group from a baseline report, with Wilson error bars and k/n."""
    summaries = [report["ungrounded"][name] for name in group_names]
    upper, lower = _error_arrays(summaries)
    pct = round(report["confidence"] * 100)
    fig = go.Figure(go.Bar(
        x=[f"{name}<br>{_kn(s)}" for name, s in zip(group_names, summaries)],
        y=[s["rate"] if s["rate"] is not None else 0 for s in summaries],
        marker_color=[TYPE_COLORS.get(name, GREY) for name in group_names],
        error_y={"type": "data", "array": upper, "arrayminus": lower, "visible": True},
        text=[_kn(s) for s in summaries], textposition="outside",
        customdata=[[name, _kn(s)] for name, s in zip(group_names, summaries)],
        hovertemplate="%{customdata[0]}: %{y:.1%} (%{customdata[1]})<extra></extra>",
    ))
    fig.update_layout(template=TEMPLATE, showlegend=False, margin={"t": 60},
                      title=f"Ungrounded rate by query type, {pct}% Wilson interval",
                      xaxis_title="query type (n judged)")
    _rate_axis(fig)
    return fig


def make_cause_figure(report: dict, group_names: list[str]) -> go.Figure:
    """Failure cause of ungrounded answers per group: rate over ALL judged answers of the group."""
    pct = round(report["confidence"] * 100)
    fig = go.Figure()
    for cause, color in CAUSE_COLORS.items():
        rates = [report["causes"][name][cause]["rate"] for name in group_names]
        upper, lower = _error_arrays(rates)
        fig.add_trace(go.Bar(
            name=cause, x=[f"{name} ({_kn(r)})" for name, r in zip(group_names, rates)],
            y=[r["rate"] if r["rate"] is not None else 0 for r in rates], marker_color=color,
            error_y={"type": "data", "array": upper, "arrayminus": lower, "visible": True},
            customdata=[[_kn(r)] for r in rates],
            hovertemplate=cause + ": %{y:.1%} (%{customdata[0]})<extra></extra>"))
    fig.update_layout(template=TEMPLATE, barmode="group", legend_title="failure cause",
                      title=f"Ungrounded answers by failure cause, {pct}% Wilson interval",
                      xaxis_title="query type (k/n of ungrounded among judged)", margin={"t": 60})
    _rate_axis(fig)
    return fig
