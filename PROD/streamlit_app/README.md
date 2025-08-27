# WWR-ML Streamlit App

Run locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Data assumptions:
- Weekly CSV at `data/weekly.csv`
- Thresholds JSON at `data/thresholds.json`
- Optional global SHAP importances at `out_explain/global_stageB_importance.csv`
