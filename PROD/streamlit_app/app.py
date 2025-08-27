# pages/6_About_Receiver_Score.py
import streamlit as st

st.set_page_config(page_title="About Receiver Score", layout="wide")
st.title("About Receiver Score")

st.markdown("""
### What is Receiver Score?
**Receiver Score** is a simple 0–100 rating that tells you **how well a pass-catcher is winning on his routes** in the window you selected (season, weeks, postseason).  
It’s **position- and season-aware** — WRs are compared to WRs, TEs to TEs, and Backs to Backs **within the same season**, so an “80” means “top-end for that position this season,” not 80 yards.

---

### How to read it
- **90–100** → *Elite* for that position/season  
- **70–89** → *Good*  
- **50–69** → *Average*  
- **30–49** → *Below Average*  
- **0–29**  → *Weak*

Use the score as a **quality signal** for routes and usage. Box scores bounce; process travels.

---

### What usually pushes the score **up**
- **Winning your route** against both **Man** and **Zone**.
- Helpful **situations**:  
  – **Motion**, **play-action (PAP)**, or **RPO**  
  – **Slot** or advantageous **alignments**  
  – A healthy **mix of depths** (Short/Intermediate/Deep) that fits the player’s role.
- **Designed involvement**: being the **first read** or **design read** on targets.

### What can pull the score **down**
- Lots of **contested targets** with low win rates (less separation).  
- **One-note deep usage** without catchable looks (boom/bust weeks without stable quality).  
- Role changes or injuries that cut into route quality.

### Things that are **contextual** (not automatically good or bad)
- **Behind-LOS/Short** usage: great for backs/YAC profiles; may cap outside WR ceilings.  
- **Light DB counts (<5 DB)**: changes spacing/matchups in ways that vary by team/role.

> Heads-up: we show **Route Rate** and **Target Share** for context, but they **do not directly drive the score**; they describe usage, not the modeled route quality itself.

---

### Why this is different from “targets” or “yards”
Those are **results**. Receiver Score focuses on the **process** that usually leads to those results — winning routes in valuable situations — so it’s helpful even when the box score is noisy.

---

### Tips for using the app
- **Leaderboards**: Sort by score to find efficient risers; then sanity-check **routes run** to gauge sample size.  
- **Player Comparison**: Compare any two players (or the same player across different weeks/seasons) to see how **score** and **drivers** shift over time.  
- **Weekly view**: Spikes happen on small route counts — look for **sustained strength**.

---

### Caveats
- **Sample size matters**: very few routes can spike a week.  
- **Context matters**: QB play, weather, injuries, and role tweaks can tilt things.

""")

with st.expander("For the Nerds (optional details)"):
    st.markdown("""
- Receiver Score comes from a **weighted route-win rate (WWR)**. Each route is weighted by the **expected fantasy value if targeted**, and wins are measured relative to **coverage and context**. We aggregate that over your selected window and put it on a **0–100 scale** using season- and position-specific bands.
- The 0–100 scale is indexed to each **season × position group** (WR / TE / Backfield) using typical cut points (think 30/50/70/90 bands). Backfield thresholds are slightly adjusted to fit RB usage patterns.
- After model tuning, we apply small **coverage/team calibrations** to remove residual bias.
""")
