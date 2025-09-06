# app.py — Streamlit Borehole Section Profile
# Interactive Plotly profile (on the web page) + Generate button + Proposed points + downloads

import io
import json
from typing import Dict, List, Tuple

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
CIRCLE_RADIUS_PX    = 10          # folium dot
CIRCLE_STROKE       = "#ffffff"
CIRCLE_STROKE_W     = 1
CIRCLE_FILL_OPACITY = 0.95
LABEL_DX_PX = 8
LABEL_DY_PX = -10

# Fixed soil colors + legend order
SOIL_COLOR_MAP = {
    "Topsoil": "#ffffcb", "SM": "#76d7c4", "SC-SM": "#fff59d", "CL": "#c5cae9",
    "PWR": "#808080", "RF": "#929591", "ML": "#ef5350", "CL-ML": "#ef9a9a",
    "CH": "#64b5f6", "MH": "#ffb74d", "GM": "#aed581", "SC": "#81c784",
    "Rock": "#f8bbd0", "SM-SC": "#e1bee7", "SP": "#ce93d8", "SW": "#ba68c8",
    "GW": "#c8e6c9", "SM-ML": "#dcedc8", "CL-CH": "#fff176", "SC-CL": "#ffee58",
}
ORDERED_SOIL_TYPES = [
    "Topsoil","SM","SC-SM","CL","PWR","RF","ML","CL-ML","CH","MH","GM",
    "SC","Rock","SM-SC","SP","SW","GW","SM-ML","CL-CH","SC-CL"
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
    if value is None:
        return "N = N/A"
    s = str(value).strip()
    if s.upper() == "N/A" or s == "":
        return "N = N/A"
    nums = []
    for x in s.split(","):
        x = x.strip().replace('"', '')
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
    # Soil type normalization
    df['Soil_Type'] = df['Soil_Type'].astype(str)
    df['Soil_Type'] = df['Soil_Type'].str.extract(r'\((.*?)\)').fillna(
        df['Soil_Type'].str.replace(r'^.*top\s*soil.*$', 'Topsoil', case=False, regex=True)
    )
    # numeric coords
    df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    needed = ['Borehole','Elevation_From','Elevation_To','Soil_Type','Latitude','Longitude','SPT']
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

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

# ── Section helpers ──────────────────────────────────────────────────────────
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

# ── Interactive Plotly profile ───────────────────────────────────────────────
def build_plotly_profile(
    df: pd.DataFrame,
    selected_bhs_ordered: List[str],
    positions_chainage: Dict[str, float],
    y_min: float,
    y_max: float,
    title: str,
    fig_height_px: int = 900,
    column_width: float = 60.0
) -> go.Figure:
    """Create a Plotly figure with rectangle shapes and text annotations."""
    half = column_width / 2
    shapes = []
    annotations = []
    used_types = set()
    unknown_types = set()

    # Collect layer rectangles + labels
    for bh in selected_bhs_ordered:
        bore_data = df[df['Borehole'] == bh]
        if bore_data.empty:
            continue
        x = positions_chainage[bh]
        top_elev = bore_data['Elevation_From'].max()
        # Borehole label above
        annotations.append(dict(
            x=x, y=top_elev + 3, text=str(bh),
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=12, color="#111", family="Arial Black")
        ))
        for _, row in bore_data.iterrows():
            elev_from = float(row['Elevation_From'])
            elev_to   = float(row['Elevation_To'])
            soil_type = str(row['Soil_Type'])
            spt_label = str(row['SPT_Label'])
            color = SOIL_COLOR_MAP.get(soil_type, "#cccccc")
            if soil_type not in SOIL_COLOR_MAP:
                unknown_types.add(soil_type)
            used_types.add(soil_type)

            # rectangle
            shapes.append(dict(
                type="rect",
                x0=x - half, x1=x + half,
                y0=elev_to, y1=elev_from,
                line=dict(color="#000000", width=1),
                fillcolor=color
            ))
            # label inside layer
            annotations.append(dict(
                x=x, y=(elev_from + elev_to)/2,
                text=f"{soil_type} ({spt_label})",
                showarrow=False,
                xanchor="center", yanchor="middle",
                font=dict(size=10, color="#111", family="Arial")
            ))

    # Build figure
    fig = go.Figure()

    # Legend (dummy traces for used soil types in fixed order)
    legend_types = [s for s in ORDERED_SOIL_TYPES if s in used_types]
    legend_types += [s for s in sorted(unknown_types) if s not in legend_types]
    for soil in legend_types:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=SOIL_COLOR_MAP.get(soil, "#cccccc")),
            name=soil, showlegend=True
        ))

    xmin = (min(positions_chainage.values()) - half) if positions_chainage else -half
    xmax = (max(positions_chainage.values()) + 15*half) if positions_chainage else half

    fig.update_layout(
        title=title,
        xaxis_title="Chainage along section (ft)",
        yaxis_title="Elevation (ft)",
        shapes=shapes,
        annotations=annotations,
        height=fig_height_px,
        margin=dict(l=60, r=260, t=60, b=60),
        plot_bgcolor="white",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, bordercolor="#ddd", borderwidth=1)
    )
    fig.update_xaxes(range=[xmin, xmax], showgrid=True, gridcolor="#eee")
    fig.update_yaxes(range=[y_min, y_max], showgrid=True, gridcolor="#eee")
    return fig

# ── Map helpers ──────────────────────────────────────────────────────────────
def add_labeled_point(fmap: folium.Map, lat: float, lon: float, name: str, text_color: str):
    folium.CircleMarker(
        location=(lat, lon),
        radius=CIRCLE_RADIUS_PX,
        color=CIRCLE_STROKE,
        weight=CIRCLE_STROKE_W,
        fill=True,
        fill_color=text_color,
        fill_opacity=CIRCLE_FILL_OPACITY,
    ).add_to(fmap)
    label_html = (
        f"<div style='background:transparent;border:none;box-shadow:none;"
        f"pointer-events:none; padding:0; margin:0;"
        f"transform: translate({LABEL_DX_PX}px, {LABEL_DY_PX}px);"
        f"display:inline-block; white-space:nowrap;"
        f"font-size:13px; font-weight:700; color:{text_color};"
        f"text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;'>"
        f"{name}</div>"
    )
    folium.map.Marker(
        location=(lat, lon),
        icon=folium.DivIcon(html=label_html, icon_size=(0,0), icon_anchor=(0,0), class_name="leaflet-empty"),
    ).add_to(fmap)

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Borehole Section Profile", layout="wide")
st.title("Soil Profile – Section View (Streamlit)")

st.sidebar.header("1) Upload data")
main_file = st.sidebar.file_uploader("MAIN borehole Excel (with Elevations/Soils/SPT)", type=["xlsx", "xls"], accept_multiple_files=False)
prop_file = st.sidebar.file_uploader("Optional: PROPOSED.xlsx (lat/lon/name)", type=["xlsx", "xls"], accept_multiple_files=False)

st.sidebar.header("2) Map & Section")
basemap_name = st.sidebar.selectbox(
    "Basemap",
    ["Esri Satellite", "OpenStreetMap", "Esri Streets", "Esri WorldTopo", "OpenTopoMap", "Carto Light", "Carto Dark", "NASA Night"],
    index=0,
)
corridor_ft = st.sidebar.number_input("Corridor (ft)", min_value=10.0, step=10.0, value=200.0)
ymin = st.sidebar.number_input("Y min (ft)", value=900.0)
ymax = st.sidebar.number_input("Y max (ft)", value=1060.0)
title = st.sidebar.text_input("Plot Title", value="Soil Profile")
figw_in = st.sidebar.number_input("Figure width (in) – ignored for web", value=100.0, step=5.0)
figh_in = st.sidebar.number_input("Figure height (in)", value=20.0, step=1.0)

if main_file is None:
    st.info("Upload your MAIN borehole Excel to begin. Optional: also upload PROPOSED.xlsx.")
    st.stop()

# Load data
try:
    df = load_df_from_excel(main_file)
except Exception as e:
    st.error(f"Failed to read MAIN Excel: {e}")
    st.stop()

bh_coords = make_borehole_coords(df)

# Proposed
proposed_df = pd.DataFrame(columns=["Latitude","Longitude","Name"])
if prop_file is not None:
    proposed_df = load_proposed_df(prop_file.getvalue())

# Center map
pts = [bh_coords[["Latitude","Longitude"]]]
if not proposed_df.empty:
    pts.append(proposed_df[["Latitude","Longitude"]])
all_pts = pd.concat(pts, ignore_index=True)
center_lat = float(all_pts['Latitude'].mean())
center_lon = float(all_pts['Longitude'].mean())

# Map + draw
def make_base_map(center_latlon=(39.0, -104.9), zoom=13) -> folium.Map:
    fmap = Map(location=center_latlon, zoom_start=zoom, control_scale=True)
    tiles = {
        "OpenStreetMap": "OpenStreetMap",
        "Esri Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Esri Streets": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "Esri WorldTopo": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "OpenTopoMap": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "Carto Light": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "Carto Dark": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "NASA Night": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_CityLights_2012/default/{time}/{tilematrixset}{maxZoom}/{z}/{y}/{x}.jpg",
    }
    attr = {
        "OpenStreetMap": "© OpenStreetMap",
        "Esri Satellite": "Tiles © Esri",
        "Esri Streets": "Tiles © Esri",
        "Esri WorldTopo": "Tiles © Esri",
        "OpenTopoMap": "© OpenTopoMap",
        "Carto Light": "© CartoDB",
        "Carto Dark": "© CartoDB",
        "NASA Night": "NASA GIBS",
    }
    for name, url in tiles.items():
        if name == "NASA Night":
            folium.raster_layers.TileLayer(tiles=url, name=name, attr=attr[name], overlay=False, control=True,
                                           **{"time": "2012-01-01", "tilematrixset": "GoogleMapsCompatible", "maxZoom": 8}).add_to(fmap)
        else:
            folium.raster_layers.TileLayer(tiles=url, name=name, attr=attr[name], overlay=False, control=True).add_to(fmap)
    LayerControl(position='topright').add_to(fmap)
    return fmap

fmap = make_base_map(center_latlon=(center_lat, center_lon), zoom=13)

# Existing (blue)
for _, r in bh_coords.iterrows():
    add_labeled_point(fmap, float(r['Latitude']), float(r['Longitude']), str(r['Borehole']), EXISTING_TEXT_COLOR)
# Proposed (red)
if not proposed_df.empty:
    for _, r in proposed_df.iterrows():
        nm = str(r.get("Name","")).strip() or "Proposed"
        add_labeled_point(fmap, float(r['Latitude']), float(r['Longitude']), nm, PROPOSED_TEXT_COLOR)

draw = Draw(
    draw_options={
        "polyline": {"shapeOptions": {"color": "#3388ff", "weight": 4}},
        "polygon": False, "circle": False, "rectangle": False, "marker": False, "circlemarker": False,
    },
    edit_options={"edit": True, "remove": True},
)
draw.add_to(fmap)

st.subheader("Map – draw your section (double-click to finish)")
map_out = st_folium(
    fmap, height=600, use_container_width=True,
    returned_objects=["all_drawings", "last_active_drawing"],
    key="section_map"
)

# Extract section line (persist across reruns)
def extract_linestring_from_map_out(mo):
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

maybe_line = extract_linestring_from_map_out(map_out or {})
if maybe_line is not None:
    st.session_state["section_line_coords"] = list(map(list, maybe_line.coords))

st.subheader("Generate Soil Profile")
go_btn = st.button("Generate Soil Profile", type="primary")

if not st.session_state["section_line_coords"]:
    st.info("Draw a polyline on the map, then click **Generate Soil Profile**.")
    st.stop()

section_line = LineString(st.session_state["section_line_coords"])

if not go_btn:
    st.info("Click **Generate Soil Profile** to compute the corridor, table, and plot.")
    st.stop()

# Compute corridor selection (EXISTING boreholes only)
def chainage_and_offset_df(bh_coords_df, line: LineString) -> pd.DataFrame:
    rows = []
    for _, r in bh_coords_df.iterrows():
        ch, off = chainage_and_offset_ft(line, float(r['Latitude']), float(r['Longitude']))
        rows.append({
            "Borehole": r["Borehole"],
            "Latitude": float(r["Latitude"]),
            "Longitude": float(r["Longitude"]),
            "Chainage_ft": round(ch, 2),
            "Offset_ft": round(off, 2),
        })
    return pd.DataFrame(rows)

report = chainage_and_offset_df(bh_coords, section_line)
sel = report[report["Offset_ft"] <= float(corridor_ft)].copy()
sel.sort_values(by=["Chainage_ft", "Borehole"], inplace=True)
sel.reset_index(drop=True, inplace=True)

if sel.empty:
    st.warning(f"No EXISTING boreholes found within {corridor_ft:.0f} ft of the drawn section.")
    st.stop()

st.markdown("**Selected boreholes within corridor**")
st.dataframe(sel, use_container_width=True)

positions_chainage = {bh: ch for bh, ch in sel[["Borehole","Chainage_ft"]].itertuples(index=False)}
selected_bhs_ordered = sel['Borehole'].tolist()

# Interactive profile on the page
fig_height_px = int(figh_in * 50)  # rough scaling inches→pixels
plot_df = df[df['Borehole'].isin(selected_bhs_ordered)]
fig = build_plotly_profile(
    df=plot_df,
    selected_bhs_ordered=selected_bhs_ordered,
    positions_chainage=positions_chainage,
    y_min=float(ymin),
    y_max=float(ymax),
    title=title,
    fig_height_px=fig_height_px
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# Downloads (CSV, GeoJSON, PNG of Plotly figure via kaleido)
csv_buf = io.StringIO()
sel.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode("utf-8")
st.download_button("Download CSV (section_boreholes.csv)", data=csv_bytes, file_name="section_boreholes.csv", mime="text/csv", use_container_width=True)

gj = {
    "type": "Feature",
    "properties": {"name": "section_line"},
    "geometry": {"type": "LineString", "coordinates": st.session_state['section_line_coords']}
}
gj_bytes = json.dumps(gj, indent=2).encode("utf-8")
st.download_button("Download section line (section_line.geojson)", data=gj_bytes, file_name="section_line.geojson", mime="application/geo+json", use_container_width=True)

try:
    # requires kaleido in requirements
    png_bytes = fig.to_image(format="png", scale=2)
    st.download_button("Download plot (soil_profile.png)", data=png_bytes, file_name="soil_profile.png", mime="image/png", use_container_width=True)
except Exception:
    st.info("PNG export needs the 'kaleido' package. Add it to requirements.txt if you want the PNG download.")
