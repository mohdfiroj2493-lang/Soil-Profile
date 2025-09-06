import io
import json
import math
import os
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geopy.distance import geodesic
from shapely.geometry import LineString, Point


import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import Map, LayerControl
from folium.plugins import Draw


# ── Visual config ────────────────────────────────────────────────────────────
EXISTING_TEXT_COLOR = "#1e88e5" # blue
PROPOSED_TEXT_COLOR = "#e53935" # red
CIRCLE_RADIUS_PX = 10 # dot size (approx for folium)
CIRCLE_STROKE = "#ffffff"
CIRCLE_STROKE_W = 1
CIRCLE_FILL_OPACITY = 0.95
LABEL_DX_PX = 8
LABEL_DY_PX = -10


# ── FIXED soil colors + ordered legend ───────────────────────────────────────
SOIL_COLOR_MAP = {
"Topsoil": "#ffffcb",
"SM": "#76d7c4",
"SC-SM": "#fff59d",
"CL": "#c5cae9",
"PWR": "#808080",
"RF": "#929591",
"ML": "#ef5350",
"CL-ML": "#ef9a9a",
"CH": "#64b5f6",
"MH": "#ffb74d",
"GM": "#aed581",
"SC": "#81c784",
"Rock": "#f8bbd0",
"SM-SC": "#e1bee7",
"SP": "#ce93d8",
"SW": "#ba68c8",
"GW": "#c8e6c9",
"SM-ML": "#dcedc8",
"CL-CH": "#fff176",
"SC-CL": "#ffee58",
}
ORDERED_SOIL_TYPES = [
"Topsoil", "SM", "SC-SM", "CL", "PWR", "RF", "ML", "CL-ML", "CH", "MH", "GM",
"SC", "Rock", "SM-SC", "SP", "SW", "GW", "SM-ML", "CL-CH", "SC-CL"
]


# ── Columns + SPT averaging ─────────────────────────────────────────────────
RENAME_MAP = {
'Bore Log': 'Borehole',
'Elevation From': 'Elevation_From',
'Elevation To': 'Elevation_To',
'Soil Layer Description': 'Soil_Type',
'Latitude': 'Latitude',
'Longitude': 'Longitude',
'SPT N-Value': 'SPT',
}


def compute_spt_avg(value):
if pd.isna(value) or str(value).strip().upper() == "N/A":
return "N = N/A"
try:
nums = [
float(x.strip().replace('"', ''))
for x in str(value).split(",")
if x.strip().replace('"', '').replace('.', '', 1).isdigit()
]
return f"N = {round(sum(nums)/len(nums), 2)}" if nums else "N = N/A"
except Exception:
return "N = N/A"


@st.cache_data(show_spinner=False)
st.caption("✅ Labels are clean text (no bubble) with white halo; dots are larger for visibility. Draw your section, set corridor/Y‑limits/title/figure size, then click Generate.")
