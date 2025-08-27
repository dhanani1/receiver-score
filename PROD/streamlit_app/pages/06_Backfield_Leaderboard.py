import streamlit as st
import plotly.graph_objects as go
from utils import load_weekly, add_weekly_rates, aggregate_and_rate
import pandas as pd

st.set_page_config(page_title="Backfield Leaderboard (RB/FB)", layout="wide")
st.title("Backfield Leaderboard (RB/FB)")

@st.cache_data
def _load():
    return load_weekly()

df = _load()
seasons = sorted([int(x) for x in df["Season"].dropna().unique().tolist()])
latest  = max(seasons) if seasons else None
weeks_all = sorted([int(x) for x in df["Week"].dropna().unique().tolist()])
teams_all = sorted(df["Team"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.header("Filters")
    season    = st.selectbox("Season", seasons, index=(seasons.index(latest) if latest in seasons else 0))
    seas_type = st.radio("Season Type", ["REG","POST"], horizontal=True, index=0)
    weeks     = st.multiselect("Weeks", weeks_all, default=weeks_all)
    teams     = st.multiselect("Offense (Team)", teams_all, default=teams_all)
    view      = st.radio("View", ["Aggregated","Weekly rows"], horizontal=True, index=0)
    apply_week_min   = st.checkbox("Apply min per-week routes", value=True)
    min_week_routes  = st.number_input("Min routes per week", min_value=0, value=10, step=1)
    apply_season_min = st.checkbox("Apply min season routes (Aggregated only)", value=True)
    min_season_routes= st.number_input("Min routes per season", min_value=0, value=100, step=1)

mask = (df["Season"].eq(season) & df["Seas_Type"].eq(seas_type) &
        df["Week"].isin(weeks) & df["db_pos"].astype(str).str.upper().isin(["RB","FB"]))
if teams: mask &= df["Team"].astype(str).isin(teams)
sub = df[mask].copy()

if view == "Aggregated":
    t = aggregate_and_rate(sub, apply_week_min=apply_week_min, min_week_routes=int(min_week_routes),
                           group_by_team=False, attach_primary_team=True)
    t = t.loc[:, ~t.columns.duplicated()]
    if apply_season_min and "routes_week" in t.columns and min_season_routes>0:
        t = t[t["routes_week"] >= min_season_routes]
    cols = [c for c in [
        "Player_Name","Team","db_pos","receiver_tier","receiver_score","routes_week","targets_week",
        "route_rate_team","target_share_team"
    ] if c in t.columns]
else:
    t = add_weekly_rates(sub).copy()
    t = t.loc[:, ~t.columns.duplicated()]
    if apply_week_min and min_week_routes>0 and "routes_week" in t.columns:
        t = t[t["routes_week"] >= min_week_routes]
    cols = [c for c in [
        "Season","Seas_Type","Week","Player_Name","Team","db_pos","receiver_tier","receiver_score",
        "routes_week","targets_week","route_rate_team","target_share_team"
    ] if c in t.columns]

t_out = t.copy()
for c in [x for x in ["route_rate_team","target_share_team"] if x in t_out.columns]:
    t_out[c] = 100.0 * t_out[c]

DISPLAY_MAP = {
    "Player_Name":"Player","db_pos":"Position",
    "receiver_tier":"Receiver Tier","receiver_score":"Receiver Score",
    "routes_week":"Routes","targets_week":"Targets",
    "route_rate_team":"Route Rate (team)","target_share_team":"Target Share (team)",
}
cols_disp = [DISPLAY_MAP.get(c, c) for c in cols]
df_show = t_out[cols].copy()
df_show.columns = cols_disp

if "Receiver Score" in df_show.columns:
    df_show = df_show.sort_values("Receiver Score", ascending=False, kind="mergesort")

cfg = {"Receiver Score": st.column_config.NumberColumn(format="%.0f"),
       "Routes": st.column_config.NumberColumn(format="%d"),
       "Targets": st.column_config.NumberColumn(format="%d")}
for c in ["Route Rate (team)","Target Share (team)"]:
    if c in df_show.columns:
        cfg[c] = st.column_config.NumberColumn(format="%.1f%%")

st.caption(f"Players: {len(df_show)}")
st.dataframe(
    df_show,
    use_container_width=True, height=640, hide_index=True, column_config=cfg
)

if view == "Aggregated" and not df_show.empty and {"Receiver Score","Routes"}.issubset(df_show.columns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_show["Receiver Score"], y=df_show["Routes"], mode="markers",
        hovertemplate="%{text}<br>Score %{x:.0f} | Routes %{y:.0f}<extra></extra>",
        text=df_show["Player"] + " — " + (df_show["Team"].astype(str) if "Team" in df_show.columns else ""),
        marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="rgba(0,0,0,0.3)"))
    ))
    fig.update_layout(title="Receiver Score vs Routes",
                      xaxis=dict(title="Receiver Score", range=[0,100], tickformat="d"),
                      yaxis=dict(title="Routes", tickformat="d"),
                      height=520, margin=dict(l=40,r=20,t=60,b=40))
    st.plotly_chart(fig, use_container_width=True)
