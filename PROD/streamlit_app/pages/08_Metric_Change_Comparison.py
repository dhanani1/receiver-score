# /08_Metric_Change_Comparison.py (updated with Routes/Route Rate columns)
import streamlit as st
import pandas as pd
import numpy as np

from utils import load_weekly, aggregate_and_rate, add_weekly_rates, receiver_score_total_from_week_slice

st.set_page_config(page_title="Metric Change Comparison", layout="wide")
st.title("Metric Change Comparison")

@st.cache_data
def _load():
    return load_weekly()

df = _load()
if df is None or df.empty:
    st.warning("weekly.csv loaded empty — nothing to show.")
    st.stop()

# ---------- selectors ----------
def weeks_for(season: int, seas_type: str) -> list[int]:
    m = (df['Season'].eq(season)) & (df['Seas_Type'].eq(seas_type))
    return sorted([int(x) for x in df.loc[m, 'Week'].dropna().unique().tolist()])

seasons   = sorted([int(x) for x in df['Season'].dropna().unique().tolist()])
latest    = max(seasons) if seasons else None
prev_season = None
if latest is not None:
    before = [s for s in seasons if s < latest]
    prev_season = max(before) if before else latest

teams_all = sorted(df['Team'].dropna().astype(str).unique().tolist())
pos_opts  = ["WR","TE","RB","FB"]

# Metric catalog (now includes RR defaults and new expected metrics)
METRICS = [
    ("Receiver Score / RR", "receiver_score_rr"),     # aggregated RR score
    ("Receiver Score - Total", "receiver_score_total"),       # aggregated RR tier (label)
    ("Aimed Target Share (team)", "target_share_team"),
    ("1st-Read Share (team)", "first_read_share_team"),
    ("Designed Reads", "designed_reads"),
    ("Man Win Rate", "man_win_rate"),
    ("Zone Win Rate", "zone_win_rate"),
    ("Slot Rate", "slot_rate"),
    ("Motion Rate", "motion_rate"),
    ("Play-Action Rate", "pap_rate"),
    ("RPO Rate", "rpo_rate"),
    ("Behind-the-LOS Rate", "behind_los_rate"),
    ("Short Rate", "short_rate"),
    ("Intermediate Rate", "intermediate_rate"),
    ("Deep Rate", "deep_rate"),
    ("<5 DB Rate", "lt5db_rate"),
    ("Horizontal Route Rate","horizontal_route_rate"),
    ("Condensed Route Rate","condensed_route_rate"),
    ("Catchable Share", "catchable_share"),
    ("Contested Share", "contested_share"),
    ("Win Rate", "win_rate"),
    # NEW expected/usage
    ("xTargets", "xTargets"),
    ("TPRR", "tprr"),
    ("xTPRR", "xTPRR"),
]
name2col = {d:c for d,c in METRICS}

PCT_METRIC_KEYS = {
    "target_share_team","first_read_share_team","designed_reads",
    "man_win_rate","zone_win_rate","slot_rate","motion_rate","pap_rate","rpo_rate",
    "behind_los_rate","short_rate","intermediate_rate","deep_rate","lt5db_rate",
    "horizontal_route_rate","condensed_route_rate",
    "catchable_share","contested_share","win_rate",
    "tprr","xTPRR",
}

with st.sidebar:
    st.header("Metric")
    metric_label = st.selectbox("Select metric", options=[m[0] for m in METRICS], index=0)
    metric_col = name2col[metric_label]

    st.divider()
    st.header("Filter Set A")
    A_season = st.selectbox("A: Season", seasons, index=(seasons.index(latest) if latest in seasons else 0))
    A_type   = st.radio("A: Season Type", ["REG","POST"], horizontal=True, index=0)
    A_weeks_all = weeks_for(A_season, A_type)
    A_weeks  = st.multiselect("A: Weeks", A_weeks_all, default=A_weeks_all, key="A_weeks")
    A_pos    = st.multiselect("A: Position", pos_opts, default=pos_opts, key="A_pos")
    A_teams  = st.multiselect("A: Offense (Team)", teams_all, default=teams_all, key="A_teams")
    A_min_rt = st.number_input("A: Min routes / week", min_value=0, value=0, step=1, key="A_min_rt")
    A_season_min_toggle = st.radio("A: Apply min routes per season?", ["Off","On"], horizontal=True, index=0, key="A_season_min_toggle")
    A_min_season_routes = st.number_input("A: Min routes per season", min_value=0, value=100, step=10,
                                          key="A_min_season_routes", disabled=(A_season_min_toggle=="Off"))

    st.divider()
    st.header("Filter Set B")
    default_B_index = seasons.index(prev_season) if prev_season in seasons else (seasons.index(latest) if latest in seasons else 0)
    B_season = st.selectbox("B: Season", seasons, index=default_B_index)
    B_type   = st.radio("B: Season Type", ["REG","POST"], horizontal=True, index=0, key="B_type")
    B_weeks_all = weeks_for(B_season, B_type)
    B_weeks  = st.multiselect("B: Weeks", B_weeks_all, default=B_weeks_all, key="B_weeks")
    B_pos    = st.multiselect("B: Position", pos_opts, default=pos_opts, key="B_pos")
    B_teams  = st.multiselect("B: Offense (Team)", teams_all, default=teams_all, key="B_teams")
    B_min_rt = st.number_input("B: Min routes / week", min_value=0, value=0, step=1, key="B_min_rt")
    B_season_min_toggle = st.radio("B: Apply min routes per season?", ["Off","On"], horizontal=True, index=0, key="B_season_min_toggle")
    B_min_season_routes = st.number_input("B: Min routes per season", min_value=0, value=100, step=10,
                                          key="B_min_season_routes", disabled=(B_season_min_toggle=="Off"))

# ---------- helpers ----------
def _safe_div(n, d):
    try:
        n = float(n or 0.0); d = float(d or 0.0)
        return np.nan if d <= 0 or np.isnan(d) else n / d
    except Exception:
        return np.nan

def _slice(season: int, seas_type: str, weeks: list[int], pos_list: list[str], teams: list[str]) -> pd.DataFrame:
    m = (df["Season"].eq(season)) & (df["Seas_Type"].eq(seas_type))
    if weeks: m &= df["Week"].isin(weeks)
    if pos_list: m &= df["db_pos"].astype(str).str.upper().isin(pos_list)
    if teams: m &= df["Team"].astype(str).isin(teams)
    return add_weekly_rates(df[m].copy())

def _primary_team(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "routes_week" not in frame.columns:
        return frame[["Player_ID"]].drop_duplicates().assign(Team=np.nan)
    grouped = (frame.groupby(["Player_ID","Team"], dropna=False)["routes_week"].sum().reset_index())
    grouped = grouped.sort_values(["Player_ID","routes_week"], ascending=[True, False])
    top = grouped.drop_duplicates(["Player_ID"])[["Player_ID","Team"]]
    return top

def compute_metric_per_player(frame: pd.DataFrame, metric: str, min_week_routes: int,
                              apply_season_min: bool, min_season_routes: int) -> pd.DataFrame:
    f = frame.copy()
    if f.empty:
        return pd.DataFrame(columns=["Player_ID","Player_Name","db_pos","Team","value","routes_sum","label","route_rate_team_val"])

    # RR Receiver Score / Tier path (aggregated per player)
    if metric in {"receiver_score_rr","receiver_tier_rr"}:
        agg = aggregate_and_rate(
            f,
            apply_week_min=(min_week_routes > 0),
            min_week_routes=int(min_week_routes),
            group_by_team=False,
            attach_primary_team=True
        )
        # routes_sum & route_rate_team from aggregated frame
        routes_col = "routes_week" if "routes_week" in agg.columns else None
        agg["routes_sum"] = pd.to_numeric(agg[routes_col], errors="coerce") if routes_col else np.nan

        # recompute route_rate_team from weekly rows per player
        # Σ routes_week / Σ team_plays_with_route (distinct player-week-team)
        pw = f.drop_duplicates(["Season","Seas_Type","Week","Team","Player_ID"])
        sums = (pw.groupby(["Player_ID"], dropna=False)
                  .agg(routes=("routes_week","sum"),
                       team_plays=("team_plays_with_route","sum"))
                  .reset_index())
        sums["route_rate_team_val"] = sums.apply(lambda r: _safe_div(r["routes"], r["team_plays"]), axis=1)
        agg = agg.merge(sums[["Player_ID","route_rate_team_val"]], on="Player_ID", how="left")

        if apply_season_min and min_season_routes and routes_col is not None:
            agg = agg[agg["routes_sum"] >= int(min_season_routes)]

        keep_cols = ["Player_ID","Player_Name","db_pos","Team","routes_sum","route_rate_team_val","receiver_score","receiver_tier"]
        for c in keep_cols:
            if c not in agg.columns: agg[c] = np.nan
        out = agg[keep_cols].copy()
        if metric == "receiver_score_rr":
            out = out.rename(columns={"receiver_score":"value"})
            out["label"] = np.nan
        else:
            out = out.rename(columns={"receiver_tier":"label"})
            out["value"] = np.nan
        return out

    
    # Receiver Score - Total (sum xFP against summed weekly tier cutpoints)
    if metric == "receiver_score_total":
        f_use = f.copy()
        if min_week_routes and "routes_week" in f_use.columns:
            f_use = f_use[f_use["routes_week"] >= int(min_week_routes)]
        if f_use.empty:
            return pd.DataFrame(columns=["Player_ID","Player_Name","db_pos","Team","value","routes_sum","label","route_rate_team_val"])

        # Σ routes per player (for Routes columns)
        routes_sum = (f_use.groupby(["Player_ID","Player_Name","db_pos"], dropna=False)["routes_week"].sum().reset_index()
                      .rename(columns={"routes_week":"routes_sum"}))

        # Route Rate (team) per player from distinct player-week-team rows
        pw = f_use.drop_duplicates(["Season","Seas_Type","Week","Team","Player_ID"])
        rr = (pw.groupby(["Player_ID"], dropna=False)
                .agg(routes=("routes_week","sum"),
                     team_plays=("team_plays_with_route","sum"))
                .reset_index())
        rr["route_rate_team_val"] = rr.apply(lambda r: _safe_div(r["routes"], r["team_plays"]), axis=1)

        # Primary team
        team_map = _primary_team(f_use)

        # Compute Receiver Score - Total for each player by slicing their rows
        vals = []
        for pid, grp in f_use.groupby("Player_ID", dropna=False):
            val = receiver_score_total_from_week_slice(grp)
            # Use first non-null name/pos
            pname = grp["Player_Name"].iloc[0] if "Player_Name" in grp.columns else pid
            dpos  = grp["db_pos"].iloc[0] if "db_pos" in grp.columns else np.nan
            vals.append({"Player_ID": pid, "Player_Name": pname, "db_pos": dpos, "value": val})
        vals = pd.DataFrame(vals)

        out = (vals.merge(routes_sum, on=["Player_ID","Player_Name","db_pos"], how="left")
                    .merge(rr[["Player_ID","route_rate_team_val"]], on="Player_ID", how="left")
                    .merge(team_map, on="Player_ID", how="left"))
        out["label"] = np.nan
        return out[["Player_ID","Player_Name","db_pos","Team","value","routes_sum","label","route_rate_team_val"]]

    # Percent/other metrics path (including xTargets / tprr / xTPRR)
    f_use = f.copy()
    if min_week_routes and "routes_week" in f_use.columns:
        f_use = f_use[f_use["routes_week"] >= int(min_week_routes)]
    if f_use.empty:
        return pd.DataFrame(columns=["Player_ID","Player_Name","db_pos","Team","value","routes_sum","label","route_rate_team_val"])

    agg = (f_use.groupby(["Player_ID","Player_Name","db_pos"], dropna=False)
            .agg(
                routes=("routes_week","sum"),
                targets=("targets_week","sum"),
                xTargets=("xTargets_Week","sum"),
                route_wins=("route_wins_week","sum"),
                team_plays=("team_plays_with_route","sum"),
                team_pa=("team_pass_attempts_week","sum"),
                team_fr=("team_first_read_attempts_week","sum"),
                team_dr=("team_design_read_attempts_week","sum"),
                man_wins=("man_wins_week","sum"),
                man_routes=("man_routes_week","sum"),
                zone_wins=("zone_wins_week","sum"),
                zone_routes=("zone_routes_week","sum"),
                slot_routes=("slot_routes_week","sum"),
                motion_routes=("motion_routes_week","sum"),
                pap_routes=("pap_routes_week","sum"),
                rpo_routes=("rpo_routes_week","sum"),
                blos_routes=("behind_los_routes_week","sum"),
                short_routes=("short_routes_week","sum"),
                inter_routes=("intermediate_routes_week","sum"),
                deep_routes=("deep_routes_week","sum"),
                lt5db_routes=("lt5db_routes_week","sum"),
                catchable_targets=("catchable_targets_week","sum"),
                contested_targets=("contested_targets_week","sum"),
                first_read_targets=("first_read_targets_week","sum"),
                design_targets=("design_targets_week","sum"),
                horizontal_routes=("horizontal_routes_week","sum"),
                plays_leq3=("plays_leq3_total_routes_week","sum"),
            ).reset_index())

    agg["routes_sum"] = pd.to_numeric(agg["routes"], errors="coerce")
    agg["route_rate_team_val"] = agg.apply(lambda r: _safe_div(r["routes"], r["team_plays"]), axis=1)

    if apply_season_min and min_season_routes:
        agg = agg[agg["routes_sum"] >= int(min_season_routes)]

    def _value(row) -> float:
        if metric == "xTargets":
            return float(row["xTargets"])
        if metric == "tprr":
            return _safe_div(row["targets"], row["routes"])
        if metric == "xTPRR":
            return _safe_div(row["xTargets"], row["routes"])
        if metric == "target_share_team":
            return _safe_div(row["targets"], row["team_pa"])
        if metric == "first_read_share_team":
            return _safe_div(row["first_read_targets"] + row["design_targets"], row["team_fr"] + row["team_dr"])
        if metric == "designed_reads":
            return _safe_div(row["design_targets"], row["routes"])
        if metric == "man_win_rate":
            return _safe_div(row["man_wins"], row["man_routes"])
        if metric == "zone_win_rate":
            return _safe_div(row["zone_wins"], row["zone_routes"])
        if metric == "slot_rate":
            return _safe_div(row["slot_routes"], row["routes"])
        if metric == "motion_rate":
            return _safe_div(row["motion_routes"], row["routes"])
        if metric == "pap_rate":
            return _safe_div(row["pap_routes"], row["routes"])
        if metric == "rpo_rate":
            return _safe_div(row["rpo_routes"], row["routes"])
        if metric == "behind_los_rate":
            return _safe_div(row["blos_routes"], row["routes"])
        if metric == "short_rate":
            return _safe_div(row["short_routes"], row["routes"])
        if metric == "intermediate_rate":
            return _safe_div(row["inter_routes"], row["routes"])
        if metric == "deep_rate":
            return _safe_div(row["deep_routes"], row["routes"])
        if metric == "lt5db_rate":
            return _safe_div(row["lt5db_routes"], row["routes"])
        if metric == "catchable_share":
            return _safe_div(row["catchable_targets"], row["targets"])
        if metric == "contested_share":
            return _safe_div(row["contested_targets"], row["targets"])
        if metric == "win_rate":
            return _safe_div(row["route_wins"], row["routes"])
        if metric == "horizontal_route_rate":
            return _safe_div(row["horizontal_routes"], row["routes"])
        if metric == "condensed_route_rate":
            return _safe_div(row["plays_leq3"], row["routes"])
        return np.nan

    team_map = _primary_team(f_use)
    agg = agg.merge(team_map, on="Player_ID", how="left")

    agg["value"] = agg.apply(_value, axis=1)
    keep = ["Player_ID","Player_Name","db_pos","Team","value","routes_sum","route_rate_team_val"]
    out = agg[keep].copy()
    out["label"] = np.nan
    return out

# ---------- Build A / B slices ----------
A = _slice(A_season, A_type, A_weeks, A_pos, A_teams)
B = _slice(B_season, B_type, B_weeks, B_pos, B_teams)

A_tbl = compute_metric_per_player(A, metric_col, int(A_min_rt),
                                  apply_season_min=(A_season_min_toggle=="On"),
                                  min_season_routes=int(A_min_season_routes))
B_tbl = compute_metric_per_player(B, metric_col, int(B_min_rt),
                                  apply_season_min=(B_season_min_toggle=="On"),
                                  min_season_routes=int(B_min_season_routes))

# Only keep players who have a VALUE (or label) for A
A_tbl = A_tbl[(pd.notna(A_tbl["value"]) | pd.notna(A_tbl["label"]))].copy()

# ---------- Join & format ----------
merged = A_tbl.merge(B_tbl, on="Player_ID", how="left", suffixes=("_A","_B"))
merged["Player"]   = merged["Player_Name_A"]
merged["Team"]     = merged["Team_A"]
merged["Position"] = merged["db_pos_A"]

# Always-present Total Routes & Route Rate (team) columns
merged["Routes — A"] = pd.to_numeric(merged["routes_sum_A"], errors="coerce").round(0)
merged["Routes — B"] = pd.to_numeric(merged["routes_sum_B"], errors="coerce").round(0)
merged["Route Rate (team) — A"] = pd.to_numeric(merged["route_rate_team_val_A"], errors="coerce")
merged["Route Rate (team) — B"] = pd.to_numeric(merged["route_rate_team_val_B"], errors="coerce")

# metric-specific handling
A_col = f"{metric_label} — A"
B_col = f"{metric_label} — B"
D_col = "Δ (A–B)"

is_pct = metric_col in PCT_METRIC_KEYS
is_categorical = (metric_col == "receiver_tier_rr")
is_count_1dec = (metric_col == "xTargets")

if is_categorical:
    out = merged[["Player","Team","Position","Routes — A","Routes — B","Route Rate (team) — A","Route Rate (team) — B"]].copy()
    out[A_col] = merged["label_A"]
    out[B_col] = merged["label_B"]
    # no delta for tier labels
    cfg = {
        "Player": st.column_config.TextColumn(),
        "Team": st.column_config.TextColumn(),
        "Position": st.column_config.TextColumn(),
        "Routes — A": st.column_config.NumberColumn(format="%d"),
        "Routes — B": st.column_config.NumberColumn(format="%d"),
        "Route Rate (team) — A": st.column_config.NumberColumn(format="%.1f%%"),
        "Route Rate (team) — B": st.column_config.NumberColumn(format="%.1f%%"),
        A_col: st.column_config.TextColumn(),
        B_col: st.column_config.TextColumn(),
    }
    # percentage formatting
    out["Route Rate (team) — A"] = out["Route Rate (team) — A"] * 100.0
    out["Route Rate (team) — B"] = out["Route Rate (team) — B"] * 100.0
    out = out.sort_values(by=["Player"], ascending=True, kind="mergesort")
else:
    merged[A_col] = pd.to_numeric(merged["value_A"], errors="coerce")
    merged[B_col] = pd.to_numeric(merged["value_B"], errors="coerce")
    merged[D_col] = merged[A_col] - merged[B_col]

    cols_show = ["Player","Team","Position","Routes — A","Routes — B","Route Rate (team) — A","Route Rate (team) — B",A_col,B_col,D_col]
    out = merged[cols_show].copy()

    # percent formatting for route rate + selected metric if needed
    out["Route Rate (team) — A"] = out["Route Rate (team) — A"] * 100.0
    out["Route Rate (team) — B"] = out["Route Rate (team) — B"] * 100.0

    cfg = {
        "Player": st.column_config.TextColumn(),
        "Team": st.column_config.TextColumn(),
        "Position": st.column_config.TextColumn(),
        "Routes — A": st.column_config.NumberColumn(format="%d"),
        "Routes — B": st.column_config.NumberColumn(format="%d"),
        "Route Rate (team) — A": st.column_config.NumberColumn(format="%.1f%%"),
        "Route Rate (team) — B": st.column_config.NumberColumn(format="%.1f%%"),
    }
    if is_pct:
        out[A_col] = out[A_col] * 100.0
        out[B_col] = out[B_col] * 100.0
        out[D_col] = out[D_col] * 100.0
        cfg[A_col] = st.column_config.NumberColumn(A_col, format="%.1f%%")
        cfg[B_col] = st.column_config.NumberColumn(B_col, format="%.1f%%")
        cfg[D_col] = st.column_config.NumberColumn(D_col, format="%.1f%%")
    else:
        if is_count_1dec:
            cfg[A_col] = st.column_config.NumberColumn(A_col, format="%.1f")
            cfg[B_col] = st.column_config.NumberColumn(B_col, format="%.1f")
            cfg[D_col] = st.column_config.NumberColumn(D_col, format="%.1f")
        else:
            cfg[A_col] = st.column_config.NumberColumn(A_col, format="%.0f")
            cfg[B_col] = st.column_config.NumberColumn(B_col, format="%.0f")
            cfg[D_col] = st.column_config.NumberColumn(D_col, format="%.0f")

    out = out.sort_values(by=[D_col, "Player"], ascending=[False, True], kind="mergesort")

st.subheader("Metric Comparison Table")
st.caption(f"Metric: **{metric_label}** — A: {A_season} {A_type} | B: {B_season} {B_type}. "
           f"Showing players with A cohort values only.")

st.dataframe(out, use_container_width=True, height=720, hide_index=True, column_config=cfg)

# ---------- CSV download ----------
def _safe_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    return s.replace(" ", "_")

csv_bytes = out.to_csv(index=False).encode("utf-8")
fname = _safe_filename(f"metric_change_comparison__{metric_label}__A_{A_season}{A_type}_vs_B_{B_season}{B_type}.csv")
st.download_button(
    label="⬇️ Download CSV",
    data=csv_bytes,
    file_name=fname,
    mime="text/csv",
    type="primary",
    use_container_width=False,
    help="Download the current table as CSV"
)
