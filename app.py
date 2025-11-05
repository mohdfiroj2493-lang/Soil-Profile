from typing import Dict, List, Tuple, Optional
import io
import math

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

PROPOSED_TEXT_COLORS = [
    "#e53935", "#8e24aa", "#3949ab", "#00897b", "#fdd835",
    "#6d4c41", "#43a047", "#fb8c00"
]

CIRCLE_RADIUS_PX    = 10
CIRCLE_STROKE       = "#ffffff"
CIRCLE_STROKE_W     = 1
CIRCLE_FILL_OPACITY = 0.95
LABEL_DX_PX, LABEL_DY_PX = 8, -10

TITLE_DEFAULT = "Soil Profile"
FIG_HEIGHT_IN = 22.0

# Soil color mapping (truncated for brevity)
SOIL_COLOR_MAP = {
    "Topsoil": "#ffffcb", "Water": "#00ffff",
    "CL": "#c5cae9","CH": "#64b5f6","ML": "#ef5350","SM": "#76d7c4",
    "SC": "#81c784","SP": "#ce93d8","Rock": "#c0c0c0"
}
ORDERED_SOIL_TYPES = list(SOIL_COLOR_MAP.keys())

RENAME_MAP = {
    'Bore Log':'Borehole','Elevation From':'Elevation_From','Elevation To':'Elevation_To',
    'Soil Layer Description':'Soil_Type','Latitude':'Latitude','Longitude':'Longitude','SPT N-Value':'SPT',
    'Elevation Water Table':'Water_Elev',
}

# ── Utility functions ────────────────────────────────────────────────────────
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
    df['Soil_Type'] = df['Soil_Type'].astype(str)
    df['Soil_Type'] = df['Soil_Type'].str.extract(r'\((.*?)\)').fillna(df['Soil_Type'])
    df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude','Longitude']).copy()
    df['SPT_Label'] = df['SPT'].apply(compute_spt_avg)
    if 'Water_Elev' in df.columns:
        df['Water_Elev'] = pd.to_numeric(df['Water_Elev'], errors='coerce')
    else:
        df['Water_Elev'] = pd.NA
    return df

@st.cache_data(show_spinner=False)
def make_borehole_coords(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('Borehole')[['Latitude','Longitude']].first().reset_index()

def normalize_cols_general(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    if "latitude" in lower: ren[lower["latitude"]] = "Latitude"
    if "longitude" in lower: ren[lower["longitude"]] = "Longitude"
    if "name" in lower: ren[lower["name"]] = "Name"
    elif "id" in lower: ren[lower["id"]] = "Name"
    df = df.rename(columns=ren)
    if "Name" not in df.columns:
        df["Name"] = [f"Proposed-{i+1}" for i in range(len(df))]
    df["Latitude"]  = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df.dropna(subset=["Latitude","Longitude"])[["Latitude","Longitude","Name"]]

@st.cache_data(show_spinner=False)
def load_proposed_multisheet(uploaded_bytes: bytes) -> Dict[str, pd.DataFrame]:
    all_sheets = pd.read_excel(io.BytesIO(uploaded_bytes), sheet_name=None)
    result: Dict[str, pd.DataFrame] = {}
    for i, (sheet, df) in enumerate(all_sheets.items()):
        if i >= 8: break
        df_norm = normalize_cols_general(df)
        if not df_norm.empty:
            df_norm["Sheet"] = sheet
            df_norm["Color"] = PROPOSED_TEXT_COLORS[i % len(PROPOSED_TEXT_COLORS)]
            result[sheet] = df_norm
    return result

# ── Geometry helpers ─────────────────────────────────────────────────────────
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

# ── NEW: Corridor polygon helper ─────────────────────────────────────────────
def add_corridor_polygon(fmap: folium.Map, line: LineString, corridor_ft: float):
    """Draw corridor buffer (ft) around a section line on the map."""
    if line is None or len(line.coords) < 2 or corridor_ft <= 0:
        return
    ft_to_deg = 1 / 364000.0  # ~1° ≈ 364,000 ft
    corridor_deg = corridor_ft * ft_to_deg
    buffer_poly = line.buffer(corridor_deg)
    coords = list(buffer_poly.exterior.coords)
    folium.Polygon(
        locations=[(lat, lon) for lon, lat in coords],
        color="#ff0000",
        weight=2,
        fill=True,
        fill_color="#ff9999",
        fill_opacity=0.25,
        popup=f"Corridor ±{corridor_ft:.0f} ft",
    ).add_to(fmap)

# ── Map label helper ─────────────────────────────────────────────────────────
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

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Borehole Section Profile", layout="wide")

st.sidebar.header("Upload")
main_file = st.sidebar.file_uploader("MAIN borehole Excel", type=["xlsx","xls"])
prop_file = st.sidebar.file_uploader("Optional PROPOSED.xlsx (multi-sheet)", type=["xlsx","xls"])

if main_file is None:
    st.title("Map with Bore Logs")
    st.info("Upload the MAIN Excel to begin.")
    st.stop()

df = load_df_from_excel(main_file)
bh_coords = make_borehole_coords(df)

proposed_dict = {}
if prop_file is not None:
    try:
        proposed_dict = load_proposed_multisheet(prop_file.getvalue())
    except Exception as e:
        st.warning(f"Could not read Proposed workbook: {e}")

# ── Map ──────────────────────────────────────────────────────────────────────
st.title("Map with Bore Logs")
if proposed_dict:
    proposed_all = pd.concat(
        [dfp[['Latitude','Longitude']] for dfp in proposed_dict.values() if not dfp.empty],
        ignore_index=True
    )
    center_lat = float(pd.concat([bh_coords[['Latitude']], proposed_all[['Latitude']]], ignore_index=True)['Latitude'].mean())
    center_lon = float(pd.concat([bh_coords[['Longitude']], proposed_all[['Longitude']]], ignore_index=True)['Longitude'].mean())
else:
    center_lat = float(bh_coords['Latitude'].mean())
    center_lon = float(bh_coords['Longitude'].mean())

fmap = Map(location=(center_lat, center_lon), zoom_start=13, control_scale=True)
folium.TileLayer("OpenStreetMap", name="OSM").add_to(fmap)
LayerControl(position='topright').add_to(fmap)

for _, r in bh_coords.iterrows():
    add_labeled_point(fmap, float(r['Latitude']), float(r['Longitude']), str(r['Borehole']), EXISTING_TEXT_COLOR)

if proposed_dict:
    for sheet, dfp in proposed_dict.items():
        if dfp.empty: continue
        color = dfp["Color"].iloc[0]
        for _, r in dfp.iterrows():
            nm = str(r.get("Name", "")).strip() or "Proposed"
            add_labeled_point(fmap, float(r['Latitude']), float(r['Longitude']), nm, color)

Draw(
    draw_options={"polyline":{"shapeOptions":{"color":"#e50000","weight":5}},
                  "polygon":False,"circle":False,"rectangle":False,"marker":False,"circlemarker":False},
    edit_options={"edit":True,"remove":True}
).add_to(fmap)

# Corridor slider (display before map)
st.title("Section / Profile (ft) — Soil")
corridor_ft = st.slider("Corridor width (ft)", min_value=0, max_value=1000, value=200, step=10)

# Add corridor band if previous line exists
if "section_line_coords" in st.session_state and st.session_state["section_line_coords"]:
    line_existing = LineString(st.session_state["section_line_coords"])
    add_corridor_polygon(fmap, line_existing, corridor_ft)

map_out = st_folium(
    fmap, height=600, use_container_width=True,
    returned_objects=["last_active_drawing","all_drawings"], key="map"
)

# Extract section line
def extract_linestring(mo) -> LineString | None:
    lad = mo.get("last_active_drawing")
    if isinstance(lad, dict):
        geom = lad.get("geometry", {})
        if geom.get("type") == "LineString" and len(geom.get("coordinates", [])) >= 2:
            return LineString(geom["coordinates"])
    if mo.get("all_drawings") and isinstance(mo["all_drawings"], dict):
        for feat in reversed(mo["all_drawings"].get("features", [])):
            geom = feat.get("geometry", {})
            if geom.get("type") == "LineString" and len(geom.get("coordinates", [])) >= 2:
                return LineString(geom["coordinates"])
    return None

if "section_line_coords" not in st.session_state:
    st.session_state["section_line_coords"] = None

maybe_line = extract_linestring(map_out or {})
if maybe_line is not None:
    st.session_state["section_line_coords"] = list(map(list, maybe_line.coords))

if not st.session_state["section_line_coords"]:
    st.info("Draw a polyline on the map (double-click to finish). The corridor will appear automatically.")
    st.stop()

section_line = LineString(st.session_state["section_line_coords"])

# ── 3) Interactive 2D profile (chainage) — with manual width option ─────────
plot_df = df[df['Borehole'].isin(ordered_bhs)]
ymin_auto, ymax_auto = auto_y_limits(plot_df)
fig_height_px = int(FIG_HEIGHT_IN * 50)

# Auto width vs manual slider
suggested = dynamic_column_width(xpos)  # based on 80% of min gap
auto_width = st.checkbox("Auto column width", value=True)
if auto_width:
    column_width_ft = None
    st.caption(f"Auto width ≈ **{suggested:.1f} ft** (80% of nearest spacing)")
else:
    minw, maxw = 8.0, 300.0
    default_val = float(min(max(suggested, 30.0), maxw))
    column_width_ft = st.slider("Column width (ft)", min_value=minw, max_value=maxw,
                                value=default_val, step=2.0)

fig2d = build_plotly_profile(
    df=plot_df, ordered_bhs=ordered_bhs, x_positions=xpos,
    y_min=ymin_auto, y_max=ymax_auto, title=TITLE_DEFAULT,
    column_width=column_width_ft,
    show_codes=show_codes, show_spt=show_spt,
    fig_height_px=fig_height_px
)

# High-res export
modebar_cfg = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "filename": "soil_profile", "scale": 4},
}
st.plotly_chart(fig2d, use_container_width=True, config=modebar_cfg)

# ── 4) Interactive 3D Borehole View (PLAN COORDS) ───────────────────────────
st.markdown("### 3D Borehole View (ft, Plan Coordinates)")
left, right = st.columns([1, 3])
with left:
    limit_to_corridor = st.checkbox("Limit to section corridor", value=True)
with right:
    vert_exag = st.slider("Vertical exaggeration (display only)", 0.5, 10.0, 2.0, 0.1)

bhs_for_3d = ordered_bhs if limit_to_corridor else \
             [bh for bh in bh_coords['Borehole'].tolist() if (df['Borehole'] == bh).any()]

sel_coords = bh_coords[bh_coords['Borehole'].isin(bhs_for_3d)]
lat0 = float(sel_coords['Latitude'].mean()); lon0 = float(sel_coords['Longitude'].mean())
xy_map = {row['Borehole']: latlon_to_local_xy_ft(float(row['Latitude']), float(row['Longitude']), lat0, lon0)
          for _, row in sel_coords.iterrows()}

plot_df3d = df[df['Borehole'].isin(bhs_for_3d)]
ymin_auto3d, ymax_auto3d = auto_y_limits(plot_df3d)

fig3d = build_3d_profile_plan(
    df=plot_df3d, selected_bhs=bhs_for_3d, xy_ft=xy_map,
    y_min=ymin_auto3d, y_max=ymax_auto3d, title=TITLE_DEFAULT,
    column_width_ft=60.0, vert_exag=vert_exag
)
st.plotly_chart(fig3d, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"scale": 3}})
