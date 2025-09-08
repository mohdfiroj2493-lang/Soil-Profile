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

# ── 2D profile builder ───────────────────────────────────────────────────────
def build_plotly_profile(
    df: pd.DataFrame, ordered_bhs: List[str], x_positions: Dict[str,float],
    y_min: float, y_max: float, title: str,
    column_width: Optional[float],
    fig_height_px: int = 900
) -> go.Figure:
    # Width is either manual (passed) or auto from spacing
    width = column_width if column_width is not None else dynamic_column_width(x_positions)
    half = width / 2.0

    # Adaptive label sizes depending on width
    def inner_font_for(w):
        if w >= 50: return 11
        if w >= 35: return 10
        if w >= 22: return 9
        if w >= 14: return 8
        return 7
    def bh_font_for(w):
        if w >= 50: return 13
        if w >= 35: return 12
        if w >= 22: return 11
        if w >= 14: return 10
        return 9

    inner_font = inner_font_for(width)
    bh_font    = bh_font_for(width)

    shapes, annotations = [], []
    used_types, unknown_types = set(), set()

    for bh in ordered_bhs:
        bore = df[df['Borehole'] == bh]
        if bore.empty:
            continue
        x = x_positions[bh]
        top_el = bore['Elevation_From'].max()
        annotations.append(dict(
            x=x, y=top_el+3, text=bh, showarrow=False,
            xanchor="center", yanchor="bottom",
            font=dict(size=bh_font, family="Arial Black", color="#111")
        ))
        for _, r in bore.iterrows():
            ef, et = float(r['Elevation_From']), float(r['Elevation_To'])
            soil = str(r['Soil_Type'])
            spt  = str(r['SPT_Label'])
            color = SOIL_COLOR_MAP.get(soil, "#cccccc")
            if soil not in SOIL_COLOR_MAP: unknown_types.add(soil)
            used_types.add(soil)
            shapes.append(dict(type="rect", x0=x-half, x1=x+half, y0=et, y1=ef,
                               line=dict(color="#000", width=1), fillcolor=color))
            annotations.append(dict(
                x=x, y=(ef+et)/2, text=f"{soil} ({spt})", showarrow=False,
                xanchor="center", yanchor="middle",
                font=dict(size=inner_font, family="Arial", color="#111")
            ))

    # Legend: preferred order first, then any extra
    ordered_present = [s for s in ORDERED_SOIL_TYPES if s in used_types]
    extra_present   = sorted([s for s in used_types if s not in set(ORDERED_SOIL_TYPES)])
    legend_types    = ordered_present + extra_present

    fig = go.Figure()
    for soil in legend_types:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=SOIL_COLOR_MAP.get(soil, "#cccccc")),
                                 name=soil, showlegend=True))

    xs = list(x_positions.values())
    xmin = (min(xs)-half) if xs else -half
    xmax = (max(xs)+3*half) if xs else half
    fig.update_layout(
        title=title, xaxis_title="Chainage along section (ft)", yaxis_title="Elevation (ft)",
        shapes=shapes, annotations=annotations, height=fig_height_px,
        margin=dict(l=60, r=260, t=60, b=60), plot_bgcolor="white",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, bordercolor="#ddd", borderwidth=1)
    )
    fig.update_xaxes(range=[xmin, xmax], showgrid=True, gridcolor="#eee")
    fig.update_yaxes(range=[y_min, y_max], showgrid=True, gridcolor="#eee")
    return fig

# ── 3D profile (PLAN COORDS) builder ────────────────────────────────────────
def build_3d_profile_plan(
    df: pd.DataFrame,
    selected_bhs: List[str],
    xy_ft: Dict[str, Tuple[float,float]],
    y_min: float,
    y_max: float,
    title: str,
    column_width_ft: float = 60.0,
    vert_exag: float = 1.0,
) -> go.Figure:
    half = column_width_ft / 2.0
    z0 = float(df['Elevation_To'].min() if not df.empty else 0.0)
    def z_scale(z):  # display-only VE
        return z0 + (z - z0) * vert_exag

    meshes: list[go.Mesh3d] = []
    used_types = set()

    label_x, label_y, label_z, label_text = [], [], [], []

    for bh in selected_bhs:
        bore = df[df['Borehole'] == bh]
        if bore.empty or bh not in xy_ft:
            continue
        x_center, y_center = xy_ft[bh]
        x0, x1 = x_center - half, x_center + half
        y0, y1 = y_center - half, y_center + half

        top_el = float(bore['Elevation_From'].max())
        label_x.append(x_center); label_y.append(y_center)
        label_z.append(z_scale(top_el) + 3); label_text.append(str(bh))

        for _, r in bore.iterrows():
            z_bot = z_scale(float(r['Elevation_To']))
            z_top = z_scale(float(r['Elevation_From']))
            soil = str(r['Soil_Type'])
            color = SOIL_COLOR_MAP.get(soil, '#cccccc')
            used_types.add(soil)

            xs = [x0, x1, x1, x0, x0, x1, x1, x0]
            ys = [y0, y0, y1, y1, y0, y0, y1, y1]
            zs = [z_bot, z_bot, z_bot, z_bot, z_top, z_top, z_top, z_top]
            i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
            j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
            k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]

            meshes.append(go.Mesh3d(
                x=xs, y=ys, z=zs, i=i, j=j, k=k,
                color=color, opacity=0.96, flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.6),
                hoverinfo='skip', showlegend=False
            ))

    fig = go.Figure(data=meshes)

    ordered_present = [s for s in ORDERED_SOIL_TYPES if s in used_types]
    extra_present   = sorted([s for s in used_types if s not in set(ORDERED_SOIL_TYPES)])
    legend_types    = ordered_present + extra_present

    for soil in legend_types:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=8, color=SOIL_COLOR_MAP.get(soil, '#cccccc')),
            name=soil, showlegend=True
        ))

    if label_text:
        fig.add_trace(go.Scatter3d(
            x=label_x, y=label_y, z=label_z, mode='text', text=label_text,
            textposition='top center', showlegend=False
        ))

    zmin_s, zmax_s = z_scale(y_min), z_scale(y_max)
    fig.update_layout(
        title=f"{title} — 3D (Plan Coordinates)",
        scene=dict(
            xaxis=dict(title='Easting (ft, local)', backgroundcolor='white', gridcolor='#eee'),
            yaxis=dict(title='Northing (ft, local)', backgroundcolor='white', gridcolor='#eee'),
            zaxis=dict(title=f'Elevation (ft, VE×{vert_exag:.2f})',
                       range=[zmin_s, zmax_s], backgroundcolor='white', gridcolor='#eee'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.4, y=1.2, z=0.8))
        ),
        height=780,
        margin=dict(l=40, r=260, t=60, b=40),
        legend=dict(yanchor='top', y=1, xanchor='left', x=1.02, bordercolor='#ddd', borderwidth=1)
    )
    return fig

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

# ── 2) Section / Profile heading + Corridor slider ──────────────────────────
st.title("Section / Profile (ft) — Soil")
corridor_ft = st.slider("Corridor width (ft)", min_value=0, max_value=1000, value=200, step=10)

if not st.session_state["section_line_coords"]:
    st.info("Draw a polyline on the map (double-click to finish). The profiles will appear below automatically.")
    st.stop()

section_line = LineString(st.session_state["section_line_coords"])

# Select boreholes within corridor
rows = []
for _, r in bh_coords.iterrows():
    ch, off = chainage_and_offset_ft(section_line, float(r['Latitude']), float(r['Longitude']))
    if off <= float(corridor_ft):
        rows.append({"Borehole": r["Borehole"], "Chainage_ft": round(ch,2)})

if not rows:
    st.warning(f"No EXISTING boreholes within {corridor_ft:.0f} ft of the drawn section.")
    st.stop()

ordered_bhs = [r["Borehole"] for r in sorted(rows, key=lambda x: (x["Chainage_ft"], x["Borehole"]))]
xpos = {r["Borehole"]: r["Chainage_ft"] for r in rows}

# ── 3) Interactive 2D profile (chainage) — with manual width option ─────────
plot_df = df[df['Borehole'].isin(ordered_bhs)]
ymin_auto, ymax_auto = auto_y_limits(plot_df)
fig_height_px = int(FIG_HEIGHT_IN * 50)

# New controls: auto width vs manual slider
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
    column_width=column_width_ft,  # None → auto; float → manual
    fig_height_px=fig_height_px
)
st.plotly_chart(fig2d, use_container_width=True, config={"displaylogo": False})

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
st.plotly_chart(fig3d, use_container_width=True, config={"displaylogo": False})
