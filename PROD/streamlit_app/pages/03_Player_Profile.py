import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_weekly,
    add_weekly_rates,
    reasons_for_subset_vs_baseline,
    aggregate_and_rate,   # Σnum/Σden → map to 0–100
)

st.set_page_config(page_title="Player Profile", layout="wide")
st.title("Player Profile")

@st.cache_data
def _load():
    return load_weekly()

df = _load()

# ---------- selectors ----------
players   = sorted(df["Player_Name"].dropna().astype(str).unique().tolist())
seasons   = sorted([int(x) for x in df["Season"].dropna().unique().tolist()])
latest    = max(seasons) if seasons else None
weeks_all = sorted([int(x) for x in df["Week"].dropna().unique().tolist()])
seas_tog  = ["REG","POST"]

with st.sidebar:
    st.header("Filters")
    player    = st.selectbox("Player", players, index=(players.index(players[0]) if players else 0))
    season    = st.selectbox("Season", seasons, index=(seasons.index(latest) if latest in seasons else 0))
    seas_type = st.radio("Season Type", seas_tog, horizontal=True, index=0)
    weeks     = st.multiselect("Weeks", weeks_all, default=weeks_all)
    min_week_routes = st.number_input("Min routes per week", min_value=0, value=0, step=1)

# slice current & prior season (same season type)
cur  = df[(df["Player_Name"].eq(player))
          & (df["Season"].eq(season))
          & (df["Seas_Type"].eq(seas_type))
          & (df["Week"].isin(weeks))].copy()

prev = df[(df["Player_Name"].eq(player))
          & (df["Season"].eq(season - 1))
          & (df["Seas_Type"].eq(seas_type))].copy()

if min_week_routes > 0 and "routes_week" in cur.columns:
    cur  = cur[cur["routes_week"]  >= min_week_routes]
if min_week_routes > 0 and "routes_week" in prev.columns:
    prev = prev[prev["routes_week"] >= min_week_routes]

cur  = cur.sort_values("Week")
prev = prev.sort_values("Week")

if cur.empty:
    st.info("No rows for the current selection.")
    st.stop()

# ---------- weekly rates for charts/table ----------
cur_rates  = add_weekly_rates(cur).copy()
prev_rates = add_weekly_rates(prev).copy()
cur_rates["win_rate"]  = pd.to_numeric(cur_rates.get("route_wins_week",0), errors="coerce") / pd.to_numeric(cur_rates.get("routes_week",0), errors="coerce")
prev_rates["win_rate"] = pd.to_numeric(prev_rates.get("route_wins_week",0), errors="coerce") / pd.to_numeric(prev_rates.get("routes_week",0), errors="coerce")

# ---------- tier-aware aggregated Receiver Score (matches Comparison/Leaderboards) ----------
prof_agg = aggregate_and_rate(cur, apply_week_min=False, group_by_team=False, attach_primary_team=False)
prof_rs  = float(prof_agg["receiver_score"].iloc[0]) if "receiver_score" in prof_agg.columns and len(prof_agg) else np.nan

# Also pull aggregates for counts
tot_routes   = float(prof_agg.get("routes_week", pd.Series([0])).iloc[0]) if "routes_week" in prof_agg.columns else pd.to_numeric(cur.get("routes_week",0), errors="coerce").sum()
tot_targets  = float(prof_agg.get("targets_week", pd.Series([0])).iloc[0]) if "targets_week" in prof_agg.columns else pd.to_numeric(cur.get("targets_week",0), errors="coerce").sum()
tot_rec      = float(prof_agg.get("receptions_week", pd.Series([0])).iloc[0]) if "receptions_week" in prof_agg.columns else pd.to_numeric(cur.get("receptions_week",0), errors="coerce").sum() if "receptions_week" in cur.columns else 0.0
tot_yards    = float(prof_agg.get("receiving_yards_week", pd.Series([0])).iloc[0]) if "receiving_yards_week" in prof_agg.columns else pd.to_numeric(cur.get("receiving_yards_week",0), errors="coerce").sum() if "receiving_yards_week" in cur.columns else 0.0

# ---------- metric catalog ----------
METRICS = [
    ("Receiver Score","receiver_score"),
    ("Win Rate","win_rate"),
    ("Routes","routes_week"),
    ("Targets","targets_week"),
    # team-denominator shares
    ("Route Rate","route_rate_team"),
    ("Target Share","target_share_team"),
    ("1st-Read Share","first_read_share_team"),
    ("Design-Read Share","design_share_team"),
    # modeled situational rates
    ("Man Win Rate","man_win_rate"),
    ("Zone Win Rate","zone_win_rate"),
    ("Slot Rate","slot_rate"),
    ("Motion Rate","motion_rate"),
    ("Play-Action Rate","pap_rate"),
    ("RPO Rate","rpo_rate"),
    ("Behind LOS Rate","behind_los_rate"),
    ("Short Rate","short_rate"),
    ("Intermediate Rate","intermediate_rate"),
    ("Deep Rate","deep_rate"),
    ("<5 DB Rate","lt5db_rate"),
    ("Catchable Share","catchable_share"),
    ("Contested Share","contested_share"),
]
name2col = {d:c for d,c in METRICS}
is_pct_label = lambda s: ("Rate" in s) or ("Share" in s) or (s == "Win Rate")

# Aggregated value for the two metric tiles:
def agg_metric_value(label: str) -> float:
    col = name2col[label]
    if col == "receiver_score":
        # Use tier-aware aggregated Receiver Score (NOT avg of weekly scores)
        return prof_rs
    if col in ["routes_week","targets_week"]:
        # For counts we show sums across the window
        return tot_routes if col == "routes_week" else tot_targets
    # For rates/shares, show the mean of weekly rates in the window
    return pd.to_numeric(cur_rates.get(col, np.nan), errors="coerce").mean()

left, right = st.columns(2)
m1 = left.selectbox("Metric (Chart 1)", options=[m[0] for m in METRICS], index=0)
m2 = right.selectbox("Metric (Chart 2)", options=[m[0] for m in METRICS], index=1)

v1 = agg_metric_value(m1)
v2 = agg_metric_value(m2)

left.metric(f"{m1} (agg)", f"{v1:.0f}" if m1=="Receiver Score" else (f"{v1*100:.1f}%" if is_pct_label(m1) else f"{v1:.0f}"))
right.metric(f"{m2} (agg)", f"{v2:.0f}" if m2=="Receiver Score" else (f"{v2*100:.1f}%" if is_pct_label(m2) else f"{v2:.0f}"))

st.caption(f"**Totals —** Routes {tot_routes:.0f} | Targets {tot_targets:.0f} | Receptions {tot_rec:.0f} | Yards {tot_yards:.0f}")

# ---------- hover explainers (vs selected-window baseline) ----------
baseline = cur_rates.copy()
hover = []
for _, r in cur_rates.iterrows():
    wk = int(r["Week"]) if pd.notna(r["Week"]) else None
    if wk is None:
        hover.append("")
        continue
    sub = cur_rates[cur_rates["Week"].eq(wk)]
    txt, _ = reasons_for_subset_vs_baseline(sub, baseline)
    hover.append(f"Week {wk} — {txt if txt else 'No prominent drivers'}")
if len(hover) != len(cur_rates):
    hover = [""]*len(cur_rates)

# ---------- plotting ----------
def plot_metric(label: str):
    col = name2col[label]
    if col not in cur_rates.columns:
        st.info(f"{label} not available.")
        return
    x_cur = pd.to_numeric(cur_rates["Week"], errors="coerce")
    y_cur = pd.to_numeric(cur_rates[col], errors="coerce")
    x_prev = pd.to_numeric(prev_rates["Week"], errors="coerce") if col in prev_rates.columns else None
    y_prev = pd.to_numeric(prev_rates[col], errors="coerce") if col in prev_rates.columns else None

    fig = go.Figure()
    # current season
    fig.add_trace(go.Scatter(
        x=x_cur, y=y_cur, mode="lines+markers", name=f"{player} {season} {seas_type}",
        text=hover,
        hovertemplate="%{text}<br>Week %{x} — Value %{y:.1%}<extra></extra>" if is_pct_label(label)
                      else "%{text}<br>Week %{x} — Value %{y:.0f}<extra></extra>"
    ))
    # prior season overlay (same seas_type)
    if x_prev is not None and y_prev is not None and len(prev_rates):
        fig.add_trace(go.Scatter(
            x=x_prev, y=y_prev, mode="lines+markers", name=f"{player} {season-1} {seas_type}",
            hovertemplate="Week %{x} — Value %{y:.1%}<extra></extra>" if is_pct_label(label)
                          else "Week %{x} — Value %{y:.0f}<extra></extra>"
        ))

    # axes
    if label == "Receiver Score":
        fig.update_yaxes(title_text=label, range=[0,100], tickformat="d")
        for y,name in [(30,"Below Avg"),(50,"Average"),(70,"Good"),(90,"Elite")]:
            fig.add_hline(y=float(y), line_dash="dot", annotation_text=name, annotation_position="top left")
    elif is_pct_label(label):
        fig.update_yaxes(title_text=label, rangemode="tozero", tickformat=".1%")
    else:
        fig.update_yaxes(title_text=label, rangemode="tozero", tickformat="d")

    fig.update_xaxes(title_text="Week")
    fig.update_layout(title=f"{player} — {label} by Week ({season} {seas_type})", height=430)
    st.plotly_chart(fig, use_container_width=True)

st.subheader(f"{player} ({season} {seas_type}) — Selected Weeks: " +
             ", ".join(map(str, sorted(set(cur_rates['Week'].dropna().astype(int).tolist())))))
st.markdown("#### Chart 1"); plot_metric(m1)
st.markdown("#### Chart 2"); plot_metric(m2)
