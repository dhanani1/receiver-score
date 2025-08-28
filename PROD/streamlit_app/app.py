import streamlit as st

st.set_page_config(page_title="About Receiver Score", page_icon="📘", layout="wide")
st.title("About Receiver Score")

st.markdown("""
### What is Receiver Score?
**Receiver Score** is a 0–100 rating that answers: *“How much scoring value did this player **create on his routes**, relative to peers, in this season?”*  

It’s **position- and season-aware** — WRs are compared to WRs, TEs to TEs, and Backs to Backs **within the same season**.  
An **80** means “top-end for that position *this season*,” not 80 yards.

---

### Why it’s different
Most numbers only reward what **already happened** (targets, yards). Receiver Score evaluates the **process** on *every* route, including those where the ball never arrived:

- Did the player **win** the route?
- **How valuable** is that situation (coverage, depth, alignment, motion, PA/RPO, personnel)?
- **How likely** is a win here to draw a target?
- **If targeted**, what’s the **expected fantasy value** (xFP) of this route?

We credit **every winning route**, even when untargeted, by assigning it counterfactual value:  
> *“In this same situation, wins are usually thrown and worth X if thrown.”*

---

### Concrete examples
- **Slot WR vs LB, wide open, no target**  
  Traditional xFP: **0**.  
  Receiver Score: recognizes **slot vs non-DB** is high leverage, **P(target)** is elevated, and **xFP if targeted** is strong → assigns credit.
- **Shallow flat vs Deep PA post**  
  A win on a shallow flat yields modest value; a win on a deep **play-action** post vs **man** is high value → two players with the same raw win% can have very different scores.
- **Usage mix matters**  
  WR-A: wins often but mostly on low-yield flats vs zone → modest score.  
  WR-B: wins less often but in **intermediate/deep**, favorable coverages, and with heavier offensive personnel (fewer routes on the field) → higher score.

---

### What **pushes the score up**
- **Beating non-DBs**: wins while aligned on an LB/Safety (e.g., slot vs LB) are high leverage.  
- **Fewer routes on the play**: less route congestion → higher **P(target)** for a winning route.  
- **Favorable offensive personnel**: heavier groupings (12/21) with you at WR often increase **P(target)** on a win.  
- **Fewer defensive DBs** (e.g., base vs nickel/dime): less coverage specialization → cleaner wins.  
- **Advantaged contexts**: man coverage wins, motion, play-action, RPO, and alignment leverage that historically convert to targets.  
- **Role-appropriate depth**: intermediate/deep wins for perimeter WRs; efficient short/intermediate wins for TEs and backs in space.

### What **pulls the score down**
- **Route diet of low-yield flats**: lots of shallow flats/throw-aways that rarely convert to meaningful xFP, even when “won.”  
- **Unfavorable coverages**: e.g., living in zones where your wins rarely draw targets or only generate short xFP.  
- **Unfavorable alignments/matchups**: repeatedly working into bracket/cloud looks, or winning in spots that rarely get thrown by your offense/QB.  
- **Crowded route concepts**: many routes on the field lowers your **P(target)**, so wins are diluted.  
- **Contested-heavy profile** without separation: wins recorded but low **catchable**/conversion expectation.

> Note: **Route Rate** and **Target Share** are shown for context, but they **do not directly drive** the score; they describe usage, not modeled route quality.

---

### How the 0–100 scale works
Scores are indexed to each **season and position** using four cut points:
- **≥ 90th percentile** → **Elite**  
- **70–89th** → **Good**  
- **50–69th** → **Average**  
- **30–49th** → **Below Average**  
- **< 30th** → **Weak**

Cut points come from **regular-season, season-level** performance (Σ wins × value / Σ value).  
To keep tiers meaningful early in a new season, we ramp the qualifier (e.g., ~10×week). In mature seasons, cut points are based on **full-time players** (e.g., ≥200 routes).

---

### How to use this app
- **Player Comparison**: compare any two windows (even different seasons). Line charts show **Receiver Score**; the **Explainer table** shows exactly *why* scores differ (motion/PA/RPO, depth, man/zone wins, 1st-read/design-read shares, etc.).  
- **Player Profile**: a single player’s weekly trend with an aggregated tile that **matches leaderboards**.  
- **Leaderboards**: sort by **Receiver Score** and sanity-check **Routes** to avoid tiny-sample spikes.  
- **Raw Data**: full table (weekly or aggregated) with **all** shares/rates; team-denominator shares are recomputed so they match comparison math.

""")

with st.expander("Under the hood (two-stage model, simplified)"):
    st.markdown(r"""
**Stage A – Pre-throw target likelihood from context**  
We estimate the **likelihood of a throw given pre-snap / pre-throw information** and a win:
- Inputs: coverage shell (man/zone), defensive DB count/personnel, offensive personnel, alignment (slot/outside), motion, play-action/RPO, leverage/matchups, etc.  
- Output: \(\;P(\text{target} \mid \text{context}, \text{win})\;\) — the chance that a winning route **would** be targeted in that situation.

**Stage B – Expected fantasy value if targeted**  
We estimate the **xFP conditional on a target** using all charted context:
- Inputs: pressure, separation, route depth/bucket, route family, catchability factors, plus the Stage-A win/throw context.  
- Output: \(\;\mathbb{E}[\text{xFP}\mid \text{target}, \text{context}]\;\) — the value of the throw if it happens.

**Per route expected value**  
\[
\underbrace{P(\text{target}\mid \text{context}, \text{win})}_{\text{Stage A}}
\times
\underbrace{\mathbb{E}[\text{xFP}\mid \text{target}, \text{context}]}_{\text{Stage B}}
\Rightarrow \text{expected contribution even if untargeted}
\]

**Window/season aggregation**  
\[
\text{WWR}_{\text{ML}}=\frac{\sum (\text{win}\times \text{expected value})}{\sum \text{expected value}}
\]
Then we map that to **0–100** using **season × position** percentiles (30/50/70/90 bands).

**Interpretation:** Receiver Score estimates *how much fantasy value the receiver **should** have generated* given **how** and **where** he won — independent of whether the QB actually threw the ball.
""")

st.info("Tip: In-season tiers tighten over time. Early weeks use a gentler route minimum for cutoffs; mature seasons use full-time thresholds so “Elite” reflects real season-long performance.")
