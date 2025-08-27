import streamlit as st
import pandas as pd
import numpy as np
from utils import load_weekly, add_weekly_rates, aggregate_and_rate

st.set_page_config(page_title="Raw Data", layout="wide")
st.title("Raw Data")

@st.cache_data
def _load():
    return load_weekly()

df = _load()
st.caption(f"Loaded weekly: rows={len(df):,} cols={len(df.columns)}")

# ---------- options / defaults ----------
seasons   = sorted([int(x) for x in df["Season"].dropna().unique().tolist()])
latest    = max(seasons) if seasons else None
weeks_all = sorted([int(x) for x in df["Week"].dropna().unique().tolist()])
teams_all = sorted(df["Team"].dropna().astype(str).unique().tolist())
pos_opts  = ["WR","TE","RB","FB"]  # HB removed

with st.sidebar:
    st.header("Filters")
    season    = st.selectbox("Season", ["All"] + seasons, index=(["All"]+seasons).index(latest if latest else "All"))
    seas_type = st.radio("Season Type", ["REG","POST","All"], horizontal=True, index=0)
    weeks     = st.multiselect("Weeks", weeks_all, default=weeks_all)
    pos_sel   = st.multiselect("Position", pos_opts, default=pos_opts)
    teams     = st.multiselect("Offense (Team)", teams_all, default=teams_all)

    # defaults you asked for
    view = st.radio("View", ["Aggregated","Weekly rows"], horizontal=True, index=0)
    apply_week_min   = st.checkbox("Apply min per-week routes", value=True)
    min_week_routes  = st.number_input("Min routes per week", min_value=0, value=10, step=1)
    apply_season_min = st.checkbox("Apply min season routes (Aggregated only)", value=True)
    min_season_routes= st.number_input("Min routes per season", min_value=0, value=100, step=1)

# ---------- slice ----------
q = pd.Series(True, index=df.index)
if season   != "All": q &= df["Season"].eq(season)
if seas_type!= "All": q &= df["Seas_Type"].eq(seas_type)
if weeks:            q &= df["Week"].isin(weeks)
if pos_sel:          q &= df["db_pos"].astype(str).str.upper().isin(pos_sel)
if teams:            q &= df["Team"].astype(str).isin(teams)
sub = df[q].copy()

# Apply per-week min routes to the weekly slice we’ll reuse for recomputation
sub_weekly = sub.copy()
if apply_week_min and min_week_routes > 0 and "routes_week" in sub_weekly.columns:
    sub_weekly = sub_weekly[sub_weekly["routes_week"] >= int(min_week_routes)]

# ---------- metric sets ----------
RATE_COLS = [
    # team denominators
    "route_rate_team","target_share_team","first_read_share_team","design_share_team",
    # route-scope rates
    "man_win_rate","zone_win_rate","slot_rate","motion_rate","pap_rate","rpo_rate",
    "behind_los_rate","short_rate","intermediate_rate","deep_rate","lt5db_rate",
    # target-scope shares
    "catchable_share","contested_share",
    # simple win rate
    "win_rate",
]
COUNT_COLS = ["routes_week","targets_week","receptions_week","receiving_yards_week"]
BASE_COLS  = ["Season","Seas_Type","Week","Team","Player_Name","db_pos","receiver_tier","receiver_score"]

# ---------- helpers ----------
def percentize_inplace(frame: pd.DataFrame, cols: list[str]):
    """
    Multiply listed columns by 100 for display.
    Robust to duplicate column labels: if frame[c] is a DataFrame, use the first column.
    """
    for c in cols:
        if c in frame.columns:
            val = frame[c]
            # If duplicates slipped in, selecting c returns a DataFrame — squeeze to 1-D
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            frame[c] = pd.to_numeric(val, errors="coerce") * 100.0

def recompute_weekly_team_shares(dfw: pd.DataFrame) -> pd.DataFrame:
    out = dfw.copy()
    if {"routes_week","team_plays_with_route"}.issubset(out.columns):
        out["route_rate_team"] = out["routes_week"] / out["team_plays_with_route"].replace(0, np.nan)
    if {"targets_week","team_pass_attempts_week"}.issubset(out.columns):
        out["target_share_team"] = out["targets_week"] / out["team_pass_attempts_week"].replace(0, np.nan)
    if {"first_read_targets_week","team_first_read_attempts_week"}.issubset(out.columns):
        out["first_read_share_team"] = out["first_read_targets_week"] / out["team_first_read_attempts_week"].replace(0, np.nan)
    if {"design_targets_week","team_design_read_attempts_week"}.issubset(out.columns):
        out["design_share_team"] = out["design_targets_week"] / out["team_design_read_attempts_week"].replace(0, np.nan)
    if {"route_wins_week","routes_week"}.issubset(out.columns):
        out["win_rate"] = out["route_wins_week"] / out["routes_week"].replace(0, np.nan)
    return out

def recompute_aggregated_team_shares_per_player(disp_agg: pd.DataFrame, weekly_source: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Σ player numerators / Σ team denominators per player (collision-safe).
    """
    if disp_agg.empty:
        return disp_agg

    wk = weekly_source.copy()

    # Σ player numerators (RENAMED to avoid collisions)
    num_by_player = (wk.groupby(["Season","Seas_Type","Player_ID"], dropna=False)
        .agg(
            routes_week_sum=("routes_week","sum"),
            targets_week_sum=("targets_week","sum"),
            first_read_targets_week_sum=("first_read_targets_week","sum"),
            design_targets_week_sum=("design_targets_week","sum"),
            route_wins_week_sum=("route_wins_week","sum"),
        ).reset_index())

    # Σ team denominators by unique team-week per player (RENAMED)
    team_denom_by_player = (wk.drop_duplicates(["Season","Seas_Type","Week","Team","Player_ID"])
        .groupby(["Season","Seas_Type","Player_ID"], dropna=False)
        .agg(
            team_plays_with_route_sum=("team_plays_with_route","sum"),
            team_pass_attempts_week_sum=("team_pass_attempts_week","sum"),
            team_first_read_attempts_week_sum=("team_first_read_attempts_week","sum"),
            team_design_read_attempts_week_sum=("team_design_read_attempts_week","sum"),
        ).reset_index())

    merged = (disp_agg
              .merge(num_by_player, on=["Season","Seas_Type","Player_ID"], how="left")
              .merge(team_denom_by_player, on=["Season","Seas_Type","Player_ID"], how="left"))

    def _sd(n, d):
        n = float(0 if pd.isna(n) else n); d = float(0 if pd.isna(d) else d)
        return np.nan if d <= 0 else n/d

    merged["route_rate_team"]       = merged.apply(lambda r: _sd(r["routes_week_sum"], r["team_plays_with_route_sum"]), axis=1)
    merged["target_share_team"]     = merged.apply(lambda r: _sd(r["targets_week_sum"], r["team_pass_attempts_week_sum"]), axis=1)
    merged["first_read_share_team"] = merged.apply(lambda r: _sd(r["first_read_targets_week_sum"], r["team_first_read_attempts_week_sum"]), axis=1)
    merged["design_share_team"]     = merged.apply(lambda r: _sd(r["design_targets_week_sum"], r["team_design_read_attempts_week_sum"]), axis=1)
    merged["win_rate"]              = merged.apply(lambda r: _sd(r["route_wins_week_sum"], r["routes_week_sum"]), axis=1)

    keep = list(disp_agg.columns) + ["route_rate_team","target_share_team","first_read_share_team","design_share_team","win_rate"]
    return merged[keep]

# ---------- build table ----------
if view == "Weekly rows":
    disp = add_weekly_rates(sub_weekly).copy()
    disp = recompute_weekly_team_shares(disp)   # exact match to Comparison logic
    disp = disp.loc[:, ~disp.columns.duplicated()]
    cols = [c for c in BASE_COLS + COUNT_COLS + RATE_COLS if c in disp.columns]

    disp_out = disp.copy()
    # de-duplicate BEFORE percentize (prevents Series/DataFrame ambiguity)
    disp_out = disp_out.loc[:, ~disp_out.columns.duplicated()]
    percentize_inplace(disp_out, [c for c in RATE_COLS if c in disp_out.columns])

else:
    # Aggregate first (tier-aware Σnum/Σden → RS), then recompute team shares per player from weekly rows
    disp = aggregate_and_rate(sub_weekly, apply_week_min=False, min_week_routes=0,
                              group_by_team=False, attach_primary_team=True)
    disp = disp.loc[:, ~disp.columns.duplicated()]

    disp = recompute_aggregated_team_shares_per_player(disp, sub_weekly)

    if apply_season_min and "routes_week" in disp.columns and min_season_routes>0:
        disp = disp[disp["routes_week"] >= min_season_routes]

    cols = [c for c in BASE_COLS + COUNT_COLS + RATE_COLS if c in disp.columns]

    disp_out = disp.copy()
    # de-duplicate BEFORE percentize here as well
    disp_out = disp_out.loc[:, ~disp_out.columns.duplicated()]
    percentize_inplace(disp_out, [c for c in RATE_COLS if c in disp_out.columns])

# ---------- normalize headers ----------
DISPLAY_MAP = {
    "Season":"Season", "Seas_Type":"Season Type", "Week":"Week", "Team":"Team",
    "Player_Name":"Player", "db_pos":"Position",
    "receiver_tier":"Receiver Tier", "receiver_score":"Receiver Score",
    "routes_week":"Routes", "targets_week":"Targets",
    "receptions_week":"Receptions", "receiving_yards_week":"Receiving Yards",
    "route_rate_team":"Route Rate (team)", "target_share_team":"Target Share (team)",
    "first_read_share_team":"1st-Read Share (team)", "design_share_team":"Design-Read Share (team)",
    "man_win_rate":"Man Win Rate", "zone_win_rate":"Zone Win Rate", "slot_rate":"Slot Rate",
    "motion_rate":"Motion Rate", "pap_rate":"Play-Action Rate", "rpo_rate":"RPO Rate",
    "behind_los_rate":"Behind-the-LOS Rate", "short_rate":"Short Rate", "intermediate_rate":"Intermediate Rate",
    "deep_rate":"Deep Rate", "lt5db_rate":"<5 DB Rate",
    "catchable_share":"Catchable Share", "contested_share":"Contested Share",
    "win_rate":"Win Rate",
}
cols = [c for c in cols if c in disp_out.columns]
cols_disp = [DISPLAY_MAP.get(c, c) for c in cols]
df_show = disp_out[cols].copy()
df_show.columns = cols_disp

# ---------- default sort by Receiver Score ----------
if "Receiver Score" in df_show.columns:
    df_show = df_show.sort_values("Receiver Score", ascending=False, kind="mergesort")

# ---------- formats ----------
cfg = {
    "Receiver Score": st.column_config.NumberColumn(format="%.0f"),
    "Routes": st.column_config.NumberColumn(format="%d"),
    "Targets": st.column_config.NumberColumn(format="%d"),
    "Receptions": st.column_config.NumberColumn(format="%d"),
    "Receiving Yards": st.column_config.NumberColumn(format="%d"),
}
percent_labels = [
    "Route Rate (team)","Target Share (team)","1st-Read Share (team)","Design-Read Share (team)",
    "Man Win Rate","Zone Win Rate","Slot Rate","Motion Rate","Play-Action Rate","RPO Rate",
    "Behind-the-LOS Rate","Short Rate","Intermediate Rate","Deep Rate","<5 DB Rate",
    "Catchable Share","Contested Share","Win Rate",
]
for lab in percent_labels:
    if lab in df_show.columns:
        cfg[lab] = st.column_config.NumberColumn(format="%.1f%%")

st.dataframe(df_show, use_container_width=True, height=720, hide_index=True, column_config=cfg)
