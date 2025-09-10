# app.py — Map → Section slider → 2D + 3D (plan) profiles
# Adds manual "Column width (ft)" slider for the 2D profile

from typing import Dict, List, Tuple, Optional
import io

import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import LineString, Point

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import Map, LayerControl
from folium.plugins import Draw
import plotly.graph_objects as go

# ── Visual config ────────────────────────────────────────────────────────────
EXISTING_TEXT_COLOR = "#1e88e5"   # blue
PROPOSED_TEXT_COLOR = "#e53935"   # red
CIRCLE_RADIUS_PX    = 10
CIRCLE_STROKE       = "#ffffff"
CIRCLE_STROKE_W     = 1
CIRCLE_FILL_OPACITY = 0.95
LABEL_DX_PX, LABEL_DY_PX = 8, -10

TITLE_DEFAULT = "Soil Profile"
FIG_HEIGHT_IN = 20.0  # inches → pixels with *50 later

# Soil colors (extend as needed)
SOIL_COLOR_MAP = {
    "Topsoil": "#ffffcb", "Water": "#00ffff",
    # Clays/Silts
    "CL": "#c5cae9","CH": "#64b5f6","CL-CH": "#fff176","CL-ML": "#ef9a9a",
    "ML": "#ef5350","MH": "#ffb74d",
    # Gravels/Sands
    "GM": "#aed581","GW": "#c8e6c9","GC": "#00ff00","GP": "#aaff32","GP-GC": "#008000","GP-GM": "#15b01a",
    "SM": "#76d7c4","SP": "#ce93d8","SC": "#81c784","SW": "#ba68c8",
    "SM-SC": "#e1bee7","SM-ML": "#dcedc8","SC-CL": "#ffee58","SC-SM": "#fff59d",
    # Rock / Fill / Weathered
    "PWR": "#808080","RF": "#929591","Rock": "#c0c0c0",
}
# Preferred legend order; anything else will be appended automatically
ORDERED_SOIL_TYPES = [
    "Topsoil", "Water",
    "SM", "SM-ML", "SM-SC", "SP", "SW",
    "SC", "SC-CL", "SC-SM",
    "CL", "CL-ML", "CL-CH", "CH",
    "ML", "MH",
    "GM", "GP-GM", "GP-GC", "GP", "GC", "GW",
    "Rock", "PWR", "RF",
]

# Column mapping + SPT label
RENAME_MAP = {
    'Bore Log':'Borehole','Elevation From':'Elevation_From','Elevation To':'Elevation_To',
    'Soil Layer Description':'Soil_Type','Latitude':'Latitude','Longitude':'Longitude','SPT N-Value':'SPT',
}
def compute_spt_avg(value):
    if value is None:
        return "N = N/A"
    s = str(value).strip()
    if s.upper() == "N/A" or s == "":
        return "N = N/A"
    nums = []
    for x in s.split(","):
        x = x.strip().replace('"','')
        try:
            nums.append(float(x))
        except ValueError:
            pass
    return f"N = {round(sum(nums)/len(nums), 2)}" if nums else "N = N/A"

@st.cache_data(show_spinner=False)
def load_df_from_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    df.rename(columns=RENAME_MAP, inplace=True)
    # normalize soil names (Topsoil) or take code in parentheses
    df['Soil_Type'] = df['Soil_Type'].astype(str)
    df['Soil_Type'] = df['Soil_Type'].str.extract(r'\((.*?)\)').fillna(
        df['Soil_Type'].str.replace(r'^.*top\s*soil.*$', 'Topsoil', case=False, regex=True)
    )
    df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude','Longitude']).copy()
    df['SPT_Label'] = df['SPT'].apply(compute_spt_avg)
    return df

@st.cache_data(show_spinner=False)
def make_borehole_coords(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('Borehole')[['Latitude','Longitude']].first().reset_index()

# Proposed points helpers
def normalize_cols_general(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    if "latitude" in lower: ren[lower["latitude"]] = "Latitude"
    elif "lat" in lower:    ren[lower["lat"]] = "Latitude"
    if "longitude" in lower: ren[lower["longitude"]] = "Longitude"
    elif "lon" in lower:     ren[lower["lon"]] = "Longitude"
    elif "long" in lower:    ren[lower["long"]] = "Longitude"
    if "name" in lower:      ren[lower["name"]] = "Name"
    elif "id" in lower:      ren[lower["id"]] = "Name"
    df = df.rename(columns=ren)
    if "Name" not in df.columns:
        df["Name"] = [f"Proposed-{i+1}" for i in range(len(df))]
    df["Latitude"]  = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df.dropna(subset=["Latitude","Longitude"])[["Latitude","Longitude","Name"]]

@st.cache_data(show_spinner=False)
def load_first_sheet_bytes(bdata: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(bdata))

@st.cache_data(show_spinner=False)
def load_proposed_df(uploaded_bytes: bytes) -> pd.DataFrame:
    try:
        df = load_first_sheet_bytes(uploaded_bytes)
        return normalize_cols_general(df)
    except Exception:
        return pd.DataFrame(columns=["Latitude","Longitude","Name"])

# ── geometry helpers ─────────────────────────────────────────────────────────
def total_geodesic_length_ft_of_linestring(line: LineString) -> float:
    coords = list(line.coords)
    total = 0.0
    for i in range(1, len(coords)):
        lon0, lat0 = coords[i-1]
        lon1, lat1 = coords[i]
        total += geodesic((lat0, lon0), (lat1, lon1)).feet
    return total

def chainage_and_offset_ft(line: LineString, lat: float, lon: float) -> Tuple[float, float]:
    if len(line.coords) < 2:
        return 0.0, 0.0
    proj = line.project(Point(lon, lat))
    frac = proj / line.length if line.length > 0 else 0.0
    total_len_ft = total_geodesic_length_ft_of_linestring(line)
    chain_ft = total_len_ft * frac
    nearest = line.interpolate(proj)
    off_ft = geodesic((lat, lon), (nearest.y, nearest.x)).feet
    return float(chain_ft), float(off_ft)

# Lat/Lon → local plan coordinates (ft), origin at (lat0,lon0)
def latlon_to_local_xy_ft(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    x_m = geodesic((lat0, lon0), (lat0, lon)).meters   # Easting
    y_m = geodesic((lat0, lon0), (lat,  lon0)).meters   # Northing
    x = x_m * 3.28084 * (1 if lon >= lon0 else -1)
    y = y_m * 3.28084 * (1 if lat >= lat0 else -1)
    return x, y

def auto_y_limits(df_subset: pd.DataFrame, pad_ratio: float = 0.05) -> Tuple[float, float]:
    if df_subset.empty:
        return 0.0, 1.0
    y_min = float(df_subset['Elevation_To'].min())
    y_max = float(df_subset['Elevation_From'].max())
    rng = max(1.0, (y_max - y_min))
    pad = rng * pad_ratio
    return y_min - pad, y_max + pad

# Auto width suggestion from spacing
def dynamic_column_width(x_positions: Dict[str, float],
                         default_width: float = 60.0,
                         min_width: float = 8.0,
                         fraction_of_min_gap: float = 0.8) -> float:
    xs = sorted(x_positions.values())
    if len(xs) < 2:
        return default_width
    gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1) if xs[i+1] > xs[i]]
    if not gaps:
        return default_width
    min_gap = min(gaps)
    width = min(default_width, fraction_of_min_gap * min_gap)
    return max(min_width, width)

# ── map helpers ──────────────────────────────────────────────────────────────
def add_labeled_point(fmap: folium.Map, lat: float, lon: float, name: str, color_hex: str):
    folium.CircleMarker(
        location=(lat, lon), radius=CIRCLE_RADIUS_PX, color=CIRCLE_STROKE,
        weight=CIRCLE_STROKE_W, fill=True, fill_color=color_hex, fill_opacity=CIRCLE_FILL_OPACITY
    ).add_to(fmap)
    label_html = (
        f"<div style='background:transparent;border:none;box-shadow:none;pointer-events:none;"
        f"padding:0;margin:0;transform: translate({LABEL_DX_PX}px, {LABEL_DY_PX}px);"
        f"display:inline-block;white-space:nowrap;font-size:13px;font-weight:700;color:{color_hex};"
        f"text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;'>{name}</div>"
    )
    folium.Marker(location=(lat, lon), icon=folium.DivIcon(html=label_html, icon_size=(0,0))).add_to(fmap)

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Borehole Section Profile", layout="wide")

# Sidebar uploads
st.sidebar.header("Upload")
main_file = st.sidebar.file_uploader("MAIN borehole Excel (Elevations/Soils/SPT)", type=["xlsx","xls"])
prop_file = st.sidebar.file_uploader("Optional PROPOSED.xlsx (lat/lon/name)", type=["xlsx","xls"])

if main_file is None:
    st.title("Map with Bore Logs")
    st.info("Upload the MAIN Excel to begin.")
    st.stop()

# Load data
df = load_df_from_excel(main_file)
bh_coords = make_borehole_coords(df)
proposed_df = pd.DataFrame(columns=["Latitude","Longitude","Name"])
if prop_file is not None:
    proposed_df = load_proposed_df(prop_file.getvalue())

# ── 1) Map with Bore Logs ───────────────────────────────────────────────────
st.title("Map with Bore Logs")

center_lat = float(pd.concat(
    [bh_coords[['Latitude']], proposed_df[['Latitude']] if not proposed_df.empty else bh_coords[['Latitude']]],
    ignore_index=True)['Latitude'].mean()
)
center_lon = float(pd.concat(
    [bh_coords[['Longitude']], proposed_df[['Longitude']] if not proposed_df.empty else bh_coords[['Longitude']]],
    ignore_index=True)['Longitude'].mean()
)

fmap = Map(location=(center_lat, center_lon), zoom_start=13, control_scale=True)
folium.raster_layers.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    name="Esri Satellite", attr="Tiles © Esri", overlay=False, control=True
).add_to(fmap)
LayerControl(position='topright').add_to(fmap)

for _, r in bh_coords.iterrows():
    add_labeled_point(fmap, float(r['Latitude']), float(r['Longitude']), str(r['Borehole']), EXISTING_TEXT_COLOR)
if not proposed_df.empty:
    for _, r in proposed_df.iterrows():
        nm = str(r.get("Name","")).strip() or "Proposed"
        add_labeled_point(fmap, float(r['Latitude']), float(r['Longitude']), nm, PROPOSED_TEXT_COLOR)

Draw(
    draw_options={"polyline":{"shapeOptions":{"color":"#3388ff","weight":4}},
                  "polygon":False,"circle":False,"rectangle":False,"marker":False,"circlemarker":False},
    edit_options={"edit":True,"remove":True}
).add_to(fmap)

map_out = st_folium(
    fmap, height=600, use_container_width=True,
    returned_objects=["last_active_drawing","all_drawings"], key="map"
)
