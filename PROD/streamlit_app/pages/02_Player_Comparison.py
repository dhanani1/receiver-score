# /02_Player_Comparison.py (updated)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_weekly,
    add_weekly_rates,
    aggregate_and_rate,
    reasons_for_subset_vs_baseline,
    receiver_score_total_from_week_slice,
)

st.set_page_config(page_title="Player Comparison", page_icon="🏈", layout="wide")
st.title("Player Comparison")

@st.cache_data
def _load():
    return load_weekly()

df = _load()

# ---------- helpers ----------
def weeks_for(season: int, seas_type: str) -> list[int]:
    mask = (df["Season"].eq(season)) & (df["Seas_Type"].eq(seas_type))
    return sorted([int(x) for x in df.loc[mask, "Week"].dropna().unique().tolist()])

def slice_player(name: str, season: int, seas_type: str, weeks: list[int], min_week_routes: int, teams: list[str] | None) -> pd.DataFrame:
    s = df[(df["Player_Name"].eq(name)) &
           (df["Season"].eq(season)) &
           (df["Seas_Type"].eq(seas_type))]
    if weeks:
        s = s[s["Week"].isin(weeks)]
    if teams:
        s = s[s["Team"].astype(str).isin(teams)]
    if min_week_routes and "routes_week" in s.columns:
        s = s[s["routes_week"] >= int(min_week_routes)]
    s = s.sort_values("Week")
    return add_weekly_rates(s)

def agg_rr_and_total(s: pd.DataFrame) -> dict:
    """Return aggregated RR score and Total score (weekly-tiers), plus counts."""
    if s.empty:
        return dict(receiver_score_rr=np.nan, receiver_score_total=np.nan,
                    routes=0, targets=0, xTargets=0.0)
    g = aggregate_and_rate(s, apply_week_min=False, group_by_team=False, attach_primary_team=False)
    rr = float(g.get("receiver_score", pd.Series([np.nan])).iloc[0]) if len(g) else np.nan
    routes = float(g.get("routes_week", pd.Series([0])).iloc[0]) if "routes_week" in g.columns else 0.0
    targets= float(g.get("targets_week", pd.Series([0])).iloc[0]) if "targets_week" in g.columns else 0.0
    xTargets = float(g.get("xTargets_Week", pd.Series([0])).iloc[0]) if "xTargets_Week" in g.columns else 0.0
    total = receiver_score_total_from_week_slice(s)
    return dict(receiver_score_rr=rr, receiver_score_total=total, routes=routes, targets=targets, xTargets=xTargets)

def plot_weekly_rr(name: str, s: pd.DataFrame):
    if s.empty:
        st.info(f"No rows for {name}."); return
    fig = go.Figure()
    y = pd.to_numeric(s.get("receiver_score_per_route", np.nan), errors="coerce")
    fig.add_trace(go.Scatter(
        x=s["Week"], y=y, mode="lines+markers", name=name,
        hovertemplate="Week %{x} — RR Score %{y:.0f}<extra></extra>",
    ))
    fig.update_yaxes(title="Receiver Score / RR", range=[0,100], tickformat="d")
    fig.update_xaxes(title="Week")
    fig.update_layout(height=360, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

def plot_weekly_total(name: str, s: pd.DataFrame):
    if s.empty:
        st.info(f"No rows for {name}."); return
    # weekly total scores already precomputed in weekly.csv under 'receiver_score'
    fig = go.Figure()
    y = pd.to_numeric(s.get("receiver_score", np.nan), errors="coerce")
    fig.add_trace(go.Scatter(
        x=s["Week"], y=y, mode="lines+markers", name=name,
        hovertemplate="Week %{x} — Total Score %{y:.0f}<extra></extra>",
    ))
    fig.update_yaxes(title="Receiver Score — Total (weekly tiers)", range=[0,100], tickformat="d")
    fig.update_xaxes(title="Week")
    fig.update_layout(height=360, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

def _sum(s: pd.Series) -> float:
    return pd.to_numeric(s, errors="coerce").sum()

def _safe_div(n, d):
    n = float(n or 0)
    d = float(d or 0)
    return np.nan if d <= 0 or np.isnan(d) else n / d

def compute_expected_stats(cohort: pd.DataFrame) -> pd.DataFrame:
    """xTargets and xTPRR for the window."""
    if cohort.empty:
        return pd.DataFrame({"Metric": ["xTargets","xTPRR"], "A": ["",""], "B": ["",""]})

    def _one(s: pd.DataFrame) -> tuple[float,float]:
        xt = _sum(s.get("xTargets_Week", 0.0))
        r  = _sum(s.get("routes_week", 0.0))
        return xt, _safe_div(xt, r)

    return _one

# ---------- independent A / B filters ----------
players = sorted(df["Player_Name"].dropna().astype(str).unique().tolist())
seasons = sorted([int(x) for x in df["Season"].dropna().unique().tolist()])
latest  = max(seasons) if seasons else None
seas_tog= ["REG","POST"]
teams_all = sorted(df["Team"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.header("Player A")
    A_player = st.selectbox("A: Player", players, index=(players.index(players[0]) if players else 0))
    A_season = st.selectbox("A: Season", seasons, index=(seasons.index(latest) if latest in seasons else 0))
    A_type   = st.radio("A: Season Type", seas_tog, horizontal=True, index=0)
    A_weeks_all = weeks_for(A_season, A_type)
    A_weeks = st.multiselect("A: Weeks", A_weeks_all, default=A_weeks_all)
    A_teams = st.multiselect("A: Offense (Team)", teams_all, default=teams_all, key="A_teams")
    A_min_rt = st.number_input("A: Min routes / week", min_value=0, value=0, step=1)

    st.divider()
    st.header("Player B")
    B_player = st.selectbox("B: Player", players, index=(players.index(players[1]) if len(players)>1 else 0))
    B_season = st.selectbox("B: Season", seasons, index=(seasons.index(latest) if latest in seasons else 0))
    B_type   = st.radio("B: Season Type", seas_tog, horizontal=True, index=0, key="B_type")
    B_weeks_all = weeks_for(B_season, B_type)
    B_weeks = st.multiselect("B: Weeks", B_weeks_all, default=B_weeks_all, key="B_weeks")
    B_teams = st.multiselect("B: Offense (Team)", teams_all, default=teams_all, key="B_teams")
    B_min_rt = st.number_input("B: Min routes / week", min_value=0, value=0, step=1, key="B_min_rt")

# ---------- build slices, agg, charts ----------
A = slice_player(A_player, A_season, A_type, A_weeks, A_min_rt, A_teams)
B = slice_player(B_player, B_season, B_type, B_weeks, B_min_rt, B_teams)

Aagg = agg_rr_and_total(A)
Bagg = agg_rr_and_total(B)

c1, c2 = st.columns(2)
with c1:
    st.subheader(f"A: {A_player} — {A_season} {A_type}")
    st.metric("Receiver Score / RR (agg)", f"{Aagg['receiver_score_rr']:.0f}" if pd.notna(Aagg['receiver_score_rr']) else "—")
    st.metric("Receiver Score — Total (agg)", f"{Aagg['receiver_score_total']:.0f}" if pd.notna(Aagg['receiver_score_total']) else "—")
    st.caption(f"Routes {Aagg['routes']:.0f} | Targets {Aagg['targets']:.0f} | xTargets {Aagg['xTargets']:.1f}")
with c2:
    st.subheader(f"B: {B_player} — {B_season} {B_type}")
    st.metric("Receiver Score / RR (agg)", f"{Bagg['receiver_score_rr']:.0f}" if pd.notna(Bagg['receiver_score_rr']) else "—")
    st.metric("Receiver Score — Total (agg)", f"{Bagg['receiver_score_total']:.0f}" if pd.notna(Bagg['receiver_score_total']) else "—")
    st.caption(f"Routes {Bagg['routes']:.0f} | Targets {Bagg['targets']:.0f} | xTargets {Bagg['xTargets']:.1f}")

st.divider()
lc, rc = st.columns(2)
with lc:
    st.markdown(f"**{A_player} — weekly Receiver Score / RR**")
    plot_weekly_rr(A_player, A)
with rc:
    st.markdown(f"**{B_player} — weekly Receiver Score / RR**")
    plot_weekly_rr(B_player, B)

lc2, rc2 = st.columns(2)
with lc2:
    st.markdown(f"**{A_player} — weekly Receiver Score — Total**")
    plot_weekly_total(A_player, A)
with rc2:
    st.markdown(f"**{B_player} — weekly Receiver Score — Total**")
    plot_weekly_total(B_player, B)

# ---------- tables ----------
def build_expected_table(A: pd.DataFrame, B: pd.DataFrame):
    def _vals(s: pd.DataFrame):
        xt = _sum(s.get("xTargets_Week", 0.0))
        rt = _sum(s.get("routes_week", 0.0))
        return xt, _safe_div(xt, rt)

    A_xT, A_xTPRR = _vals(A)
    B_xT, B_xTPRR = _vals(B)
    rows = [
        ("xTargets", A_xT, B_xT, A_xT - B_xT if pd.notna(A_xT) and pd.notna(B_xT) else np.nan),
        ("xTPRR",   A_xTPRR, B_xTPRR, A_xTPRR - B_xTPRR if pd.notna(A_xTPRR) and pd.notna(B_xTPRR) else np.nan),
    ]
    t = pd.DataFrame(rows, columns=["Metric","A","B","Δ (A–B)"])
    # format
    t.loc[t["Metric"].eq("xTargets"), ["A","B","Δ (A–B)"]] = t.loc[t["Metric"].eq("xTargets"), ["A","B","Δ (A–B)"]].applymap(
        lambda x: "" if pd.isna(x) else f"{float(x):.1f}"
    )
    t.loc[t["Metric"].eq("xTPRR"), ["A","B","Δ (A–B)"]] = t.loc[t["Metric"].eq("xTPRR"), ["A","B","Δ (A–B)"]].applymap(
        lambda x: "" if pd.isna(x) else f"{float(x)*100:.1f}%"
    )
    return t

st.divider()
st.subheader("Explainer (percent metrics)")
baseline_A = A.copy()
hoverA = []
for wk in A["Week"]:
    sub = A[A["Week"].eq(wk)]
    t, _ = reasons_for_subset_vs_baseline(sub, baseline_A)
    hoverA.append(t)

baseline_B = B.copy()
hoverB = []
for wk in B["Week"]:
    sub = B[B["Week"].eq(wk)]
    t, _ = reasons_for_subset_vs_baseline(sub, baseline_B)
    hoverB.append(t)

# Explainer and Stats from prior version remain the same behaviourally, so we reuse the same builder
def _sum(s: pd.Series) -> float:
    return pd.to_numeric(s, errors="coerce").sum()

def _safe_div(n, d):
    n = float(n or 0)
    d = float(d or 0)
    return np.nan if d <= 0 or np.isnan(d) else n / d

def compute_all_metrics(cohort: pd.DataFrame) -> dict:
    if cohort.empty:
        return {k: np.nan for k in [
            "Receptions","Targets","Receiving Yards","Routes","receiver_score",
            "route_rate_team","target_share_team","tprr",
            "first_read_share_team","design_share_team",
            "man_win_rate","zone_win_rate","slot_rate","motion_rate","pap_rate","rpo_rate",
            "behind_los_rate","short_rate","intermediate_rate","deep_rate","lt5db_rate",
            "catchable_share","contested_share","win_rate",
            "horizontal_route_rate","condensed_route_rate",
        ]}
    routes = _sum(cohort.get("routes_week", 0))
    targets = _sum(cohort.get("targets_week", 0))
    receptions = _sum(cohort.get("receptions_week", 0))
    rec_yards = _sum(cohort.get("receiving_yards_week", 0))
    t_plays = _sum(cohort.get("team_plays_with_route", 0))
    t_pa    = _sum(cohort.get("team_pass_attempts_week", 0))
    t_fr    = _sum(cohort.get("team_first_read_attempts_week", 0))
    t_dr    = _sum(cohort.get("team_design_read_attempts_week", 0))
    fr_tgts = _sum(cohort.get("first_read_targets_week", 0))
    dr_tgts = _sum(cohort.get("design_targets_week", 0))
    man_wins = _sum(cohort.get("man_wins_week", 0))
    man_rts  = _sum(cohort.get("man_routes_week", 0))
    zone_wins= _sum(cohort.get("zone_wins_week", 0))
    zone_rts = _sum(cohort.get("zone_routes_week", 0))
    slot_rts = _sum(cohort.get("slot_routes_week", 0))
    motion_rts = _sum(cohort.get("motion_routes_week", 0))
    pap_rts    = _sum(cohort.get("pap_routes_week", 0))
    rpo_rts    = _sum(cohort.get("rpo_routes_week", 0))
    blos_rts   = _sum(cohort.get("behind_los_routes_week", 0))
    short_rts  = _sum(cohort.get("short_routes_week", 0))
    inter_rts  = _sum(cohort.get("intermediate_routes_week", 0))
    deep_rts   = _sum(cohort.get("deep_routes_week", 0))
    lt5db_rts  = _sum(cohort.get("lt5db_routes_week", 0))
    route_wins = _sum(cohort.get("route_wins_week", 0))
    catchable  = _sum(cohort.get("catchable_targets_week", 0))
    contested  = _sum(cohort.get("contested_targets_week", 0))
    horiz_rts  = _sum(cohort.get("horizontal_routes_week", 0))
    plays_leq3 = _sum(cohort.get("plays_leq3_total_routes_week", 0))

    route_rate_team       = _safe_div(routes, t_plays)
    target_share_team     = _safe_div(targets, t_pa)
    first_read_share_team = _safe_div(fr_tgts + dr_tgts, t_fr + t_dr)
    design_share_team     = _safe_div(dr_tgts, routes)

    man_win_rate = _safe_div(man_wins, man_rts)
    zone_win_rate = _safe_div(zone_wins, zone_rts)
    slot_rate = _safe_div(slot_rts, routes)
    motion_rate = _safe_div(motion_rts, routes)
    pap_rate = _safe_div(pap_rts, routes)
    rpo_rate = _safe_div(rpo_rts, routes)
    behind_los_rate = _safe_div(blos_rts, routes)
    short_rate = _safe_div(short_rts, routes)
    intermediate_rate = _safe_div(inter_rts, routes)
    deep_rate = _safe_div(deep_rts, routes)
    lt5db_rate = _safe_div(lt5db_rts, routes)
    catchable_share = _safe_div(catchable, targets)
    contested_share = _safe_div(contested, targets)
    win_rate = _safe_div(route_wins, routes)
    horizontal_route_rate = _safe_div(horiz_rts, routes)
    condensed_route_rate  = _safe_div(plays_leq3, routes)

    # RR Receiver Score aggregate via aggregator
    agg = aggregate_and_rate(cohort, apply_week_min=False, group_by_team=False, attach_primary_team=False)
    receiver_score = float(agg.get("receiver_score", pd.Series([np.nan])).iloc[0]) if len(agg) else np.nan
    tprr = _safe_div(targets, routes)

    return {
        "Receptions": receptions,
        "Targets": targets,
        "Receiving Yards": rec_yards,
        "Routes": routes,
        "receiver_score": receiver_score,
        "route_rate_team": route_rate_team,
        "target_share_team": target_share_team,
        "tprr": tprr,
        "first_read_share_team": first_read_share_team,
        "design_share_team": design_share_team,
        "man_win_rate": man_win_rate,
        "zone_win_rate": zone_win_rate,
        "slot_rate": slot_rate,
        "motion_rate": motion_rate,
        "pap_rate": pap_rate,
        "rpo_rate": rpo_rate,
        "behind_los_rate": behind_los_rate,
        "short_rate": short_rate,
        "intermediate_rate": intermediate_rate,
        "deep_rate": deep_rate,
        "lt5db_rate": lt5db_rate,
        "catchable_share": catchable_share,
        "contested_share": contested_share,
        "win_rate": win_rate,
        "horizontal_route_rate": horizontal_route_rate,
        "condensed_route_rate": condensed_route_rate,
    }

def build_tables(A: pd.DataFrame, B: pd.DataFrame):
    a = compute_all_metrics(A)
    b = compute_all_metrics(B)

    expl_metrics = [
        ("1st-Read Share (team)", "first_read_share_team"),
        ("Design-Read Share (team)", "design_share_team"),
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
        ("Horizontal Route Rate", "horizontal_route_rate"),
        ("Condensed Route Rate", "condensed_route_rate"),
        ("Catchable Share", "catchable_share"),
        ("Contested Share", "contested_share"),
        ("Win Rate", "win_rate"),
    ]
    rows_expl = [(lab, a[k], b[k], (a[k]-b[k]) if (pd.notna(a[k]) and pd.notna(b[k])) else np.nan)
                 for lab,k in expl_metrics]
    df_expl = pd.DataFrame(rows_expl, columns=["Metric","A","B","Δ (A–B)"])
    df_expl[["A","B","Δ (A–B)"]] = df_expl[["A","B","Δ (A–B)"]].astype(float) * 100.0

    stat_metrics = [
        ("Receptions","Receptions"),
        ("Targets","Targets"),
        ("Receiving Yards","Receiving Yards"),
        ("Routes","Routes"),
        ("Receiver Score / RR","receiver_score"),
        ("Route Rate (team)","route_rate_team"),
        ("Target Share (team)","target_share_team"),
        ("TPRR","tprr"),
    ]
    rows_stats = [(lab, a[k], b[k], (a[k]-b[k]) if (pd.notna(a[k]) and pd.notna(b[k])) else np.nan)
                  for lab,k in stat_metrics]
    df_stats = pd.DataFrame(rows_stats, columns=["Metric","A","B","Δ (A–B)"])

    count_or_score = {"Receptions","Targets","Receiving Yards","Routes","Receiver Score / RR"}
    for m in count_or_score:
        mask = df_stats["Metric"].eq(m)
        df_stats.loc[mask, ["A","B","Δ (A–B)"]] = df_stats.loc[mask, ["A","B","Δ (A–B)"]].applymap(
            lambda x: "" if pd.isna(x) else f"{float(x):.0f}"
        )
    for m in ["Route Rate (team)","Target Share (team)","TPRR"]:
        mask = df_stats["Metric"].eq(m)
        df_stats.loc[mask, ["A","B","Δ (A–B)"]] = (df_stats.loc[mask, ["A","B","Δ (A–B)"]]
            .astype(float).applymap(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%"))

    return df_expl, df_stats

# ---------- Explainer + Stats + Expected ----------
expl_tbl, stats_tbl = build_tables(A, B)
st.dataframe(
    expl_tbl, use_container_width=True, height=560, hide_index=True,
    column_config={
        "Metric": st.column_config.TextColumn(),
        "A": st.column_config.NumberColumn("A", format="%.1f%%"),
        "B": st.column_config.NumberColumn("B", format="%.1f%%"),
        "Δ (A–B)": st.column_config.NumberColumn("Δ (A–B)", format="%.1f%%"),
    }
)

st.subheader("Expected Stats")
exp_tbl = build_expected_table(A, B)
st.dataframe(
    exp_tbl, use_container_width=True, height=160, hide_index=True,
    column_config={
        "Metric": st.column_config.TextColumn(),
        "A": st.column_config.TextColumn(),
        "B": st.column_config.TextColumn(),
        "Δ (A–B)": st.column_config.TextColumn(),
    }
)

st.subheader("Stats (counts, score, and team shares)")
st.dataframe(
    stats_tbl, use_container_width=True, height=420, hide_index=True,
    column_config={
        "Metric": st.column_config.TextColumn(),
        "A": st.column_config.TextColumn("A"),
        "B": st.column_config.TextColumn("B"),
        "Δ (A–B)": st.column_config.TextColumn("Δ (A–B)"),
    }
)
