
# Borehole Visualization Tool (Streamlit Cloud Ready)

This is your earlier 'usapp-style' app adapted for Streamlit Cloud with Python 3.13–compatible wheels.
No `shapely` is required, so builds won't try to compile GEOS.

## Deploy
1. Push these files to a GitHub repo.
2. In Streamlit Cloud, set **Main file** to `app.py` (repo root) and **Branch** to your main branch.
3. Click **Delete and redeploy** if you're replacing an existing app.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
