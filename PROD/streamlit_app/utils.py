# /Users/Zain/fantasy_wwr/PROD/streamlit_app/utils.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

# ---------- Paths ----------
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent          # /Users/Zain/fantasy_wwr
DATA_DIR     = PROJECT_ROOT / "data"
WEEKLY_CSV   = DATA_DIR / "weekly.csv"
TIERS_CSV    = DATA_DIR / "receiver_score_tiers_by_season_pos.csv"

# ---------- Loaders ----------
def load_weekly() -> pd.DataFrame:
    """
    Hardened loader; normalizes common columns and dtypes.
    """
    if not WEEKLY_CSV.exists():
        raise FileNotFoundError(
            f"weekly.csv not found at {WEEKLY_CSV}. Expected: {WEEKLY_CSV}"
        )
    df = pd.read_csv(WEEKLY_CSV, low_memory=False)

    # core ids
    for c in ("Season", "Week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    if "Seas_Type" in df.columns:
        df["Seas_Type"] = df["Seas_Type"].astype(str).str.upper().str.strip()
    else:
        df["Seas_Type"] = "REG"

    if "Player_Name" in df.columns:
        df["Player_Name"] = df["Player_Name"].astype(str)
        df = df[df["Player_Name"].str.strip().ne("").fillna(False)]

    # numerics we aggregate
    numeric_cols = [
        "WWR_ML_numerator","WWR_ML_denominator","WWR_ML_value",
        "routes_week","targets_week","route_wins_week",
        "receptions_week","receiving_yards_week",
        # team denominators
        "team_plays_with_route","team_pass_attempts_week",
        "team_first_read_attempts_week","team_design_read_attempts_week",
        # rate numerators / denominators
        "man_wins_week","zone_wins_week","man_routes_week","zone_routes_week",
        "slot_routes_week","motion_routes_week","pap_routes_week","rpo_routes_week",
        "behind_los_routes_week","short_routes_week","intermediate_routes_week","deep_routes_week",
        "lt5db_routes_week",
        "first_read_targets_week","design_targets_week",
        "catchable_targets_week","contested_targets_week",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_tiers() -> Optional[pd.DataFrame]:
    """
    Loads per-season, per-pos_group tier cutoffs (p30/p50/p70/p90) used to map
    aggregated WWR to 0–100 Receiver Score.
    """
    if not TIERS_CSV.exists():
        return None
    t = pd.read_csv(TIERS_CSV)
    # tolerate older column name
    if "tier_p30" not in t.columns and "tier_p25" in t.columns:
        t["tier_p30"] = t["tier_p25"]
    if "Season" in t.columns:
        t["Season"] = pd.to_numeric(t["Season"], errors="coerce").astype("Int64")
    return t

# ---------- Helpers ----------
def map_pos_group(db_pos: str | float | None) -> Optional[str]:
    if pd.isna(db_pos):
        return None
    p = str(db_pos).strip().upper()
    if p == "WR": return "WR"
    if p == "TE": return "TE"
    if p in ("RB","HB","FB"): return "Backfield"
    return None

def safe_rate(n, d) -> float:
    try:
        n = float(n or 0); d = float(d or 0)
        return (n/d) if d > 0 else np.nan
    except Exception:
        return np.nan

def piecewise_score(v: float, t30: float, t50: float, t70: float, t90: float) -> float:
    """
    Map WWR (0..1) to 0..100 using piecewise bands with 30/50/70/90 cutpoints.
    """
    arr = [v, t30, t50, t70, t90]
    if any(pd.isna(a) for a in arr):
        return np.nan
    eps = 1e-9
    t50 = max(t50, t30 + eps)
    t70 = max(t70, t50 + eps)
    t90 = max(t90, t70 + eps)

    if v < max(t30, eps):    # 0 → 30
        return max(0.0, 30.0 * (v / max(t30, eps)))
    if v < t50:              # 30 → 50
        return 30.0 + 20.0 * (v - t30) / (t50 - t30)
    if v < t70:              # 50 → 70
        return 50.0 + 20.0 * (v - t50) / (t70 - t50)
    if v < t90:              # 70 → 90
        return 70.0 + 20.0 * (v - t70) / (t90 - t70)
    # ≥ p90 → 90..100 (stretch last band)
    band = max(t90 - t70, eps)
    return min(100.0, 90.0 + 10.0 * (v - t90) / band)

# ---------- Weekly derived rates for table/plots ----------
RATE_DEFS: List[Tuple[str,str,str]] = [
    ("man_win_rate",  "man_wins_week",  "man_routes_week"),
    ("zone_win_rate", "zone_wins_week", "zone_routes_week"),
    ("slot_rate",     "slot_routes_week","routes_week"),
    ("motion_rate",   "motion_routes_week","routes_week"),
    ("pap_rate",      "pap_routes_week","routes_week"),
    ("rpo_rate",      "rpo_routes_week","routes_week"),
    ("behind_los_rate","behind_los_routes_week","routes_week"),
    ("short_rate",    "short_routes_week","routes_week"),
    ("intermediate_rate","intermediate_routes_week","routes_week"),
    ("deep_rate",     "deep_routes_week","routes_week"),
    ("lt5db_rate",    "lt5db_routes_week","routes_week"),
    ("catchable_share","catchable_targets_week","targets_week"),
    ("contested_share","contested_targets_week","targets_week"),
]

def add_weekly_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    f = df.copy()
    for out, num, den in RATE_DEFS:
        if num in f.columns and den in f.columns:
            f[out] = f.apply(lambda r: safe_rate(r.get(num,0), r.get(den,0)), axis=1)
    # Weekly team-denominator shares (optional; for “Weekly rows” tables)
    if {"routes_week","team_plays_with_route"}.issubset(f.columns):
        f["route_rate_team"] = f.apply(lambda r: safe_rate(r["routes_week"], r["team_plays_with_route"]), axis=1)
    if {"targets_week","team_pass_attempts_week"}.issubset(f.columns):
        f["target_share_team"] = f.apply(lambda r: safe_rate(r["targets_week"], r["team_pass_attempts_week"]), axis=1)
    if {"first_read_targets_week","team_first_read_attempts_week"}.issubset(f.columns):
        f["first_read_share_team"] = f.apply(lambda r: safe_rate(r["first_read_targets_week"], r["team_first_read_attempts_week"]), axis=1)
    if {"design_targets_week","team_design_read_attempts_week"}.issubset(f.columns):
        f["design_share_team"] = f.apply(lambda r: safe_rate(r["design_targets_week"], r["team_design_read_attempts_week"]), axis=1)
    return f

# ---------- Aggregation to PLAYER (Σnums / Σdens) + tier-indexed Receiver Score ----------
def aggregate_and_rate(
    frame: pd.DataFrame,
    apply_week_min: bool = False,
    min_week_routes: int = 0,
    group_by_team: bool = False,
    attach_primary_team: bool = True,
    recompute_receiver_score_from_WWR: bool = True,
) -> pd.DataFrame:
    """
    Aggregates the provided slice to PLAYER (or PLAYER+TEAM) and computes:
      • season_wwr = Σ WWR_ML_numerator / Σ WWR_ML_denominator
      • receiver_score = piecewise(season_wwr; season+pos_group tiers)
      • team shares: route_rate_team / target_share_team /
                     first_read_share_team / design_share_team
      • standard rates: man/zone/slot/motion/... (sum(num)/sum(den))
    """
    if frame is None or frame.empty:
        return frame.copy()

    f = frame.copy()
    if apply_week_min and min_week_routes and "routes_week" in f.columns:
        f = f[f["routes_week"] >= int(min_week_routes)]

    keys = ["Season","Seas_Type","Player_ID","Player_Name","db_pos"]
    if group_by_team:
        keys += ["Team"]

    # collect columns to sum
    sum_cols = set([
        "WWR_ML_numerator","WWR_ML_denominator",
        "routes_week","targets_week",
        "receptions_week","receiving_yards_week",
        # team denominators for shares
        "team_plays_with_route","team_pass_attempts_week",
        "team_first_read_attempts_week","team_design_read_attempts_week",
    ])
    for _, num, den in RATE_DEFS:
        sum_cols.add(num); sum_cols.add(den)
    sum_cols = [c for c in sum_cols if c in f.columns]

    g = f.groupby(keys, dropna=False)[sum_cols].sum(numeric_only=True).reset_index()

    # recompute standard rates
    for out, num, den in RATE_DEFS:
        if num in g.columns and den in g.columns:
            g[out] = g.apply(lambda r: safe_rate(r.get(num,0), r.get(den,0)), axis=1)

    # team-denominator shares
    g["route_rate_team"]       = g.apply(lambda r: safe_rate(r.get("routes_week",0), r.get("team_plays_with_route",0)), axis=1)
    g["target_share_team"]     = g.apply(lambda r: safe_rate(r.get("targets_week",0), r.get("team_pass_attempts_week",0)), axis=1)
    g["first_read_share_team"] = g.apply(lambda r: safe_rate(r.get("first_read_targets_week",0), r.get("team_first_read_attempts_week",0)), axis=1)
    g["design_share_team"]     = g.apply(lambda r: safe_rate(r.get("design_targets_week",0), r.get("team_design_read_attempts_week",0)), axis=1)

    # ΣWWR and tier-indexed Receiver Score
    den = pd.to_numeric(g.get("WWR_ML_denominator",0), errors="coerce")
    num = pd.to_numeric(g.get("WWR_ML_numerator",0), errors="coerce")
    g["season_wwr"] = np.where(den > 0, num/den, np.nan)

    tiers = load_tiers()
    if tiers is not None:
        g["pos_group"] = g["db_pos"].apply(map_pos_group)
        g = g.merge(tiers, on=["Season","pos_group"], how="left")
        g["receiver_score"] = [
            piecewise_score(v, t30, t50, t70, t90)
            for v, t30, t50, t70, t90 in zip(
                g["season_wwr"],
                g.get("tier_p30"), g.get("tier_p50"), g.get("tier_p70"), g.get("tier_p90")
            )
        ]
        def _tier(v,t30,t50,t70,t90):
            if pd.isna(v) or any(pd.isna([t30,t50,t70,t90])): return "Unknown"
            if v >= t90: return "Elite"
            if v >= t70: return "Good"
            if v >= t50: return "Average"
            if v >= t30: return "Below Average"
            return "Weak"
        g["receiver_tier"] = [
            _tier(v,t30,t50,t70,t90)
            for v,t30,t50,t70,t90 in zip(
                g["season_wwr"], g.get("tier_p30"), g.get("tier_p50"), g.get("tier_p70"), g.get("tier_p90")
            )
        ]
    else:
        # fallback: still return something sensible
        g["receiver_score"] = g["season_wwr"] * 100.0
        g["receiver_tier"] = g["receiver_score"].apply(
            lambda s: "Elite" if s>=90 else "Good" if s>=70 else "Average" if s>=50 else "Below Average" if s>=30 else "Weak"
        )

    # attach primary Team (by routes) when not grouping by team
    if attach_primary_team and (not group_by_team) and {"routes_week","Team"}.issubset(f.columns):
        k = [k for k in keys if k != "Team"]
        tr = f.groupby(k + ["Team"], dropna=False)["routes_week"].sum().reset_index()
        tr = tr.sort_values(k + ["routes_week"], ascending=[True]*len(k) + [False])
        top = tr.drop_duplicates(k)[k + ["Team"]]
        g = g.merge(top, on=k, how="left")

    return g.loc[:, ~g.columns.duplicated()]

# ---------- Comparison explainer (drivers; TEAM denominators where applicable) ----------
DRIVERS: List[Tuple[str,str,str]] = [
    ("Route rate",                 "routes_week",              "team_plays_with_route"),
    ("Target share",               "targets_week",             "team_pass_attempts_week"),
    ("1st-read share",             "first_read_targets_week",  "team_first_read_attempts_week"),
    ("Design-read share",          "design_targets_week",      "team_design_read_attempts_week"),
    ("Behind LOS rate",            "behind_los_routes_week",   "routes_week"),
    ("Short rate",                 "short_routes_week",        "routes_week"),
    ("Intermediate rate",          "intermediate_routes_week", "routes_week"),
    ("Deep rate",                  "deep_routes_week",         "routes_week"),
    ("Man rate",                   "man_routes_week",          "routes_week"),
    ("Zone rate",                  "zone_routes_week",         "routes_week"),
    ("Slot rate",                  "slot_routes_week",         "routes_week"),
    ("Motion rate",                "motion_routes_week",       "routes_week"),
    ("Play-action rate",           "pap_routes_week",          "routes_week"),
    ("RPO rate",                   "rpo_routes_week",          "routes_week"),
    ("vs <5 DB rate",              "lt5db_routes_week",        "routes_week"),
    ("Catchable share (targeted)", "catchable_targets_week",   "targets_week"),
    ("Contested share (targeted)", "contested_targets_week",   "targets_week"),
]

def _rate(n, d) -> float:
    if d is None or d == 0 or pd.isna(d):
        return np.nan
    return float(np.clip(n/d, 0, 1))

def _driver_table(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, num_col, den_col in DRIVERS:
        num = pd.to_numeric(cohort.get(num_col, 0), errors="coerce").sum()
        den = pd.to_numeric(cohort.get(den_col, 0), errors="coerce").sum()
        rows.append({"label": label, "num": float(num), "den": float(den), "rate": _rate(num, den)})
    return pd.DataFrame(rows)

def reasons_for_subset_vs_baseline(
    subset: pd.DataFrame,
    baseline: pd.DataFrame,
    min_pp: float = 5.0,
    fallback_top_k: int = 3
) -> tuple[str, pd.DataFrame]:
    """
    Returns short text and a table of the largest rate deltas (percentage points)
    between the subset and its baseline (same player-season, all selected weeks).
    """
    if subset.empty or baseline.empty:
        return "", pd.DataFrame(columns=["label","delta_pp","score","subset_rate","baseline_rate"])
    A = _driver_table(subset)
    B = _driver_table(baseline)
    M = A.merge(B, on="label", suffixes=("_A","_B"))
    M["delta"] = M["rate_A"] - M["rate_B"]
    M["delta_pp"] = (M["delta"] * 100.0).round(1)
    M["score"] = M["delta"].abs()

    sig = M[M["delta_pp"].abs() >= float(min_pp)].sort_values("score", ascending=False)
    S = sig if not sig.empty else M[M["delta_pp"] != 0.0].sort_values("score", ascending=False).head(fallback_top_k)

    if S.empty:
        return "", pd.DataFrame(columns=["label","delta_pp","score","subset_rate","baseline_rate"])

    def fmt(r): return f"{r['label']} {'↑' if r['delta']>0 else '↓'}{r['delta_pp']:.1f}pp"
    text = "; ".join(fmt(r) for _, r in S.iterrows())
    out = S[["label","delta_pp","score","rate_A","rate_B"]] \
            .rename(columns={"rate_A":"subset_rate","rate_B":"baseline_rate"}) \
            .reset_index(drop=True)
    return text, out

# ---------- Simple cohort aggregator for top cards (optional) ----------
AGG_SUMS = [
    "WWR_ML_numerator","WWR_ML_denominator","routes_week","targets_week",
    "receptions_week","receiving_yards_week",
    "team_plays_with_route","team_pass_attempts_week",
    "team_first_read_attempts_week","team_design_read_attempts_week",
]

def _aggregate_cohort(df: pd.DataFrame) -> Dict[str, float]:
    """
    For quick top cards: returns Σ totals and a Receiver Score aggregate mapped to tiers
    when Season and pos_group are unambiguous; else falls back to WWR×100.
    """
    if df is None or df.empty:
        return {k: np.nan for k in AGG_SUMS + ["WWR_ML_value","receiver_score_agg"]}
    sums = df[AGG_SUMS].sum(numeric_only=True)
    num = float(sums.get("WWR_ML_numerator",0)); den = float(sums.get("WWR_ML_denominator",0))
    wwr = (num/den) if den>0 else np.nan
    out = sums.to_dict(); out["WWR_ML_value"] = wwr

    tiers = load_tiers()
    if pd.notna(wwr) and tiers is not None:
        seasons = [s for s in pd.unique(df["Season"]) if pd.notna(s)]
        posg    = [map_pos_group(x) for x in pd.unique(df["db_pos"]) if pd.notna(x)]
        if len(seasons) == 1 and len(set(posg)) == 1:
            s  = int(seasons[0]); pg = list(set(posg))[0]
            row = tiers[(tiers["Season"].eq(s)) & (tiers["pos_group"].astype(str).eq(pg))]
            if not row.empty:
                t30, t50, t70, t90 = [float(row.iloc[0][c]) for c in ("tier_p30","tier_p50","tier_p70","tier_p90")]
                out["receiver_score_agg"] = piecewise_score(wwr, t30, t50, t70, t90)
                return out
    # fallback
    out["receiver_score_agg"] = wwr*100.0 if pd.notna(wwr) else np.nan
    return out
