import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_weekly,
    add_weekly_rates,
    aggregate_and_rate,
    reasons_for_subset_vs_baseline,
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

def slice_player(name: str, season: int, seas_type: str, weeks: list[int], min_week_routes: int) -> pd.DataFrame:
    s = df[(df["Player_Name"].eq(name)) &
           (df["Season"].eq(season)) &
           (df["Seas_Type"].eq(seas_type)) &
           (df["Week"].isin(weeks))].copy()
    if min_week_routes and "routes_week" in s.columns:
        s = s[s["routes_week"] >= int(min_week_routes)]
    s = s.sort_values("Week")
    return add_weekly_rates(s)

def tier_aware_agg(s: pd.DataFrame) -> dict:
    """Σ WWR parts → tier-indexed receiver_score; also return sums for counts."""
    if s.empty:
        return dict(receiver_score=np.nan, routes=0, targets=0, rec=0, yds=0,
                    num=0.0, den=0.0)
    g = aggregate_and_rate(s, apply_week_min=False, group_by_team=False, attach_primary_team=False)
    rs  = float(g.get("receiver_score", pd.Series([np.nan])).iloc[0])
    routes = float(g.get("routes_week", pd.Series([0])).iloc[0])
    targets= float(g.get("targets_week", pd.Series([0])).iloc[0])
    rec    = float(g.get("receptions_week", pd.Series([0])).iloc[0]) if "receptions_week" in g.columns else 0.0
    yds    = float(g.get("receiving_yards_week", pd.Series([0])).iloc[0]) if "receiving_yards_week" in g.columns else 0.0
    num    = float(g.get("WWR_ML_numerator", pd.Series([0.0])).iloc[0])
    den    = float(g.get("WWR_ML_denominator", pd.Series([0.0])).iloc[0])
    return dict(receiver_score=rs, routes=routes, targets=targets, rec=rec, yds=yds, num=num, den=den)

def plot_weekly(name: str, s: pd.DataFrame):
    if s.empty:
        st.info(f"No rows for {name}."); return
    baseline = s.copy()
    hover = []
    for wk in s["Week"]:
        sub = s[s["Week"].eq(wk)]
        t, _ = reasons_for_subset_vs_baseline(sub, baseline)
        hover.append(f"Week {int(wk)} — {t if t else 'No prominent drivers'}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["Week"], y=s["receiver_score"], mode="lines+markers", name=name,
        text=hover, hovertemplate="%{text}<br>Week %{x} — Score %{y:.0f}<extra></extra>",
    ))
    fig.update_yaxes(title="Receiver Score", range=[0,100], tickformat="d")
    fig.update_xaxes(title="Week")
    fig.update_layout(height=360, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

def _sum(s: pd.Series) -> float:
    return pd.to_numeric(s, errors="coerce").sum()

def _safe_div(n, d):
    n = float(n or 0)
    d = float(d or 0)
    return np.nan if d <= 0 or np.isnan(d) else n / d

def compute_all_metrics(cohort: pd.DataFrame) -> dict:
    """Return explainer & stats metrics for a cohort (player-window)."""
    if cohort.empty:
        return {k: np.nan for k in [
            # Stats
            "Receptions","Targets","Receiving Yards","Routes","receiver_score",
            "route_rate_team","target_share_team","tprr",
            # Explainer
            "first_read_share_team","design_share_team",
            "man_win_rate","zone_win_rate","slot_rate","motion_rate","pap_rate","rpo_rate",
            "behind_los_rate","short_rate","intermediate_rate","deep_rate","lt5db_rate",
            "catchable_share","contested_share","win_rate",
            "horizontal_route_rate","condensed_route_rate",
        ]}

    # Counts
    routes = _sum(cohort.get("routes_week", 0))
    targets = _sum(cohort.get("targets_week", 0))
    receptions = _sum(cohort.get("receptions_week", 0))
    rec_yards = _sum(cohort.get("receiving_yards_week", 0))

    # Team denominators
    t_plays = _sum(cohort.get("team_plays_with_route", 0))
    t_pa    = _sum(cohort.get("team_pass_attempts_week", 0))
    t_fr    = _sum(cohort.get("team_first_read_attempts_week", 0))
    t_dr    = _sum(cohort.get("team_design_read_attempts_week", 0))

    # Numerators
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
    # NEW numerators for the two explainer rates
    horiz_rts  = _sum(cohort.get("horizontal_routes_week", 0))
    plays_leq3 = _sum(cohort.get("plays_leq3_total_routes_week", 0))

    # Shares & rates
    route_rate_team       = _safe_div(routes, t_plays)
    target_share_team     = _safe_div(targets, t_pa)
    first_read_share_team = _safe_div(fr_tgts, t_fr)
    design_share_team     = _safe_div(dr_tgts, t_dr)

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

    # NEW: the two explainer rates you asked for
    horizontal_route_rate = _safe_div(horiz_rts, routes)
    condensed_route_rate  = _safe_div(plays_leq3, routes)

    # Tier-aware Receiver Score
    agg = aggregate_and_rate(cohort, apply_week_min=False, group_by_team=False, attach_primary_team=False)
    receiver_score = float(agg.get("receiver_score", pd.Series([np.nan])).iloc[0]) if len(agg) else np.nan

    # NEW: TPRR for Stats = targets / routes
    tprr = _safe_div(targets, routes)

    return {
        # Stats table fields
        "Receptions": receptions,
        "Targets": targets,
        "Receiving Yards": rec_yards,
        "Routes": routes,
        "receiver_score": receiver_score,
        "route_rate_team": route_rate_team,
        "target_share_team": target_share_team,
        "tprr": tprr,
        # Explainer fields
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

    # Explainer metrics (all %)
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
        # NEW in Explainer:
        ("Horizontal Route Rate", "horizontal_route_rate"),
        ("Condensed Route Rate", "condensed_route_rate"),
        ("Catchable Share", "catchable_share"),
        ("Contested Share", "contested_share"),
        ("Win Rate", "win_rate"),
    ]
    rows_expl = [(lab, a[k], b[k], (a[k]-b[k]) if (pd.notna(a[k]) and pd.notna(b[k])) else np.nan)
                 for lab,k in expl_metrics]
    df_expl = pd.DataFrame(rows_expl, columns=["Metric","A","B","Δ (A–B)"])
    # Explainer: all percents → ×100
    df_expl[["A","B","Δ (A–B)"]] = df_expl[["A","B","Δ (A–B)"]].astype(float) * 100.0

    # Stats metrics (counts + route/target share + receiver score + TPRR)
    stat_metrics = [
        ("Receptions","Receptions"),
        ("Targets","Targets"),
        ("Receiving Yards","Receiving Yards"),
        ("Routes","Routes"),
        ("Receiver Score","receiver_score"),
        ("Route Rate (team)","route_rate_team"),   # %
        ("Target Share (team)","target_share_team"),# %
        ("TPRR","tprr"),                            # NEW %
    ]
    rows_stats = [(lab, a[k], b[k], (a[k]-b[k]) if (pd.notna(a[k]) and pd.notna(b[k])) else np.nan)
                  for lab,k in stat_metrics]
    df_stats = pd.DataFrame(rows_stats, columns=["Metric","A","B","Δ (A–B)"])

    # Format: counts & score as ints; shares & TPRR as percents
    count_or_score = {"Receptions","Targets","Receiving Yards","Routes","Receiver Score"}
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

# ---------- independent A / B filters ----------
players = sorted(df["Player_Name"].dropna().astype(str).unique().tolist())
seasons = sorted([int(x) for x in df["Season"].dropna().unique().tolist()])
latest  = max(seasons) if seasons else None
seas_tog= ["REG","POST"]

with st.sidebar:
    st.header("Player A")
    A_player = st.selectbox("A: Player", players, index=(players.index("A.J. Brown") if "A.J. Brown" in players else 0))
    A_season = st.selectbox("A: Season", seasons, index=(seasons.index(latest) if latest in seasons else 0))
    A_type   = st.radio("A: Season Type", seas_tog, horizontal=True, index=0)
    A_weeks_all = weeks_for(A_season, A_type)
    A_weeks = st.multiselect("A: Weeks", A_weeks_all, default=A_weeks_all)
    A_min_rt = st.number_input("A: Min routes / week", min_value=0, value=0, step=1)

    st.divider()
    st.header("Player B")
    B_player = st.selectbox("B: Player", players, index=(players.index("CeeDee Lamb") if "CeeDee Lamb" in players else 1))
    B_season = st.selectbox("B: Season", seasons, index=(seasons.index(latest) if latest in seasons else 0))
    B_type   = st.radio("B: Season Type", seas_tog, horizontal=True, index=0, key="B_type")
    B_weeks_all = weeks_for(B_season, B_type)
    B_weeks = st.multiselect("B: Weeks", B_weeks_all, default=B_weeks_all, key="B_weeks")
    B_min_rt = st.number_input("B: Min routes / week", min_value=0, value=0, step=1, key="B_min_rt")

# ---------- build slices, agg, charts ----------
A = slice_player(A_player, A_season, A_type, A_weeks, A_min_rt)
B = slice_player(B_player, B_season, B_type, B_weeks, B_min_rt)
Aagg, Bagg = tier_aware_agg(A), tier_aware_agg(B)

c1, c2 = st.columns(2)
with c1:
    st.subheader(f"A: {A_player} — {A_season} {A_type}")
    st.metric("Receiver Score (agg)", f"{Aagg['receiver_score']:.0f}" if pd.notna(Aagg['receiver_score']) else "—")
    st.caption(f"Routes {Aagg['routes']:.0f} | Targets {Aagg['targets']:.0f} | Receptions {Aagg['rec']:.0f} | Yards {Aagg['yds']:.0f}")
with c2:
    st.subheader(f"B: {B_player} — {B_season} {B_type}")
    st.metric("Receiver Score (agg)", f"{Bagg['receiver_score']:.0f}" if pd.notna(Bagg['receiver_score']) else "—")
    st.caption(f"Routes {Bagg['routes']:.0f} | Targets {Bagg['targets']:.0f} | Receptions {Bagg['rec']:.0f} | Yards {Bagg['yds']:.0f}")

st.divider()
lc, rc = st.columns(2)
with lc:
    st.markdown(f"**{A_player} — weekly Receiver Score (hover: drivers vs baseline)**")
    plot_weekly(A_player, A)
with rc:
    st.markdown(f"**{B_player} — weekly Receiver Score (hover: drivers vs baseline)**")
    plot_weekly(B_player, B)

# ---------- tables ----------
st.divider()
st.subheader("Explainer (percent metrics)")
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

st.subheader("Stats (counts, score, and team shares)")
# Values are already formatted per-row (ints for counts/score; percents for shares/TPRR)
st.dataframe(
    stats_tbl, use_container_width=True, height=420, hide_index=True,
    column_config={
        "Metric": st.column_config.TextColumn(),
        "A": st.column_config.TextColumn("A"),
        "B": st.column_config.TextColumn("B"),
        "Δ (A–B)": st.column_config.TextColumn("Δ (A–B)"),
    }
)
