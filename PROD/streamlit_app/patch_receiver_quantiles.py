
from pathlib import Path
import re

player_path = Path("pages/1_Player.py")
src = player_path.read_text()

new_block = r"""
fig = go.Figure()
fig.add_trace(go.Scatter(x=sub["Week"], y=sub["receiver_score"], mode="lines+markers", name="Receiver Score (weekly)"))

# Dynamic quantiles by db_pos across the full dataset
df_all = load_weekly()
bucket_mask = df_all["db_pos"].astype(str).str.upper().str.startswith(pos_bucket(db_pos))
scores = pd.to_numeric(df_all[bucket_mask]["receiver_score"], errors="coerce").dropna()
if not scores.empty:
    q = scores.quantile([0.25,0.5,0.75,0.9]).to_dict()
    for y,name in [(q.get(0.25), "Weak (p25)"), (q.get(0.5), "Average (p50)"), (q.get(0.75), "Good (p75)"), (q.get(0.9), "Elite (p90)")]:
        if y is not None:
            fig.add_hline(y=float(y), line_dash="dot", annotation_text=name, annotation_position="top left")

fig.update_layout(title=f"{player} — Receiver Score by Week ({season} {seas_type})",
                  xaxis_title="Week", yaxis_title="Receiver Score",
                  height=450)

st.plotly_chart(fig, use_container_width=True)
"""

patched = re.sub(r"fig = go\.Figure\(\)[\s\S]*?st\.plotly_chart\(fig, use_container_width=True\)", new_block, src)
player_path.write_text(patched)
print("Patched pages/1_Player.py to use receiver_score quantiles.")
