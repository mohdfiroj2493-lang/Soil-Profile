# app.py — Streamlit Borehole Section Profile (with Generate button + persistence)
import io
import json
from typing import Dict, List, Tuple

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
EXISTING_TEXT_COLOR = "#1e88e5"   # blue
PROPOSED_TEXT_COLOR = "#e53935"   # red
CIRCLE_RADIUS_PX    = 10          # dot size (approx for folium)
CIRCLE_STROKE       = "#ffffff"
CIRCLE_STROKE_W     = 1
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
    """Return label like 'N = 12.5' from comma-separated SPT list or 'N = N/A'."""
    if value is None:
        return "N = N/A"
    s = str(value).strip()
    if s.upper() == "N/A" or s == "":
        return "N = N/A"
    try:
        nums = []
        for x in s.split(","):
            x = x.strip().replace('"', '')
            try:
                nums.append(float(x))
            except ValueError:
                pass
        return f"N = {round(sum(nums)/len(nums), 2)}" if nums else "N = N/A"
    except Exception:
        return "N = N/A"

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

# ── Proposed-only loader (optional helper if you add a second upload later) ──
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

# ── Profile plot ─────────────────────────────────────────────────────────────
def plot_soil_profile_original_style(
    df: pd.DataFrame,
    selected_bhs_ordered: List[str],
    positions_chainage: Dict[str, float],
    y_min: float = 900,
    y_max: float = 1060,
    title: str = "Soil Profile",
    figsize: Tuple[float, float] = (100, 20)
):
    soil_data_elev = {}
    for bh in selected_bhs_ordered:
        bore_data = df[df['Borehole'] == bh]
        if bore_data.empty:
            continue
        soil_data_elev[bh] = list(zip(
            bore_data['Elevation_From'],
            bore_data['Elevation_To'],
            bore_data['Soil_Type'],
            bore_data['SPT_Label'],
        ))

    fig, ax = plt.subplots(figsize=figsize)
    column_width = 60
    half_width = column_width / 2

    used_types = set()
    unknown_types = set()

    for bh in selected_bhs_ordered:
        if bh not in soil_data_elev:
            continue
        x = positions_chainage[bh]
        for elev_from, elev_to, soil_type, spt_label in soil_data_elev[bh]:
            color = SOIL_COLOR_MAP.get(soil_type, "#cccccc")  # default grey
            if soil_type not in SOIL_COLOR_MAP:
                unknown_types.add(soil_type)
            ax.add_patch(
                mpatches.Rectangle(
                    (x - half_width, elev_to),
                    column_width,
                    elev_from - elev_to,
                    facecolor=color,
                    edgecolor='black'
                )
            )
            ax.text(x, (elev_from + elev_to)/2, f"{soil_type} ({spt_label})",
                    ha='center', va='center', fontsize=9, weight='bold')
            used_types.add(soil_type)

    # Borehole labels at top
    for bh in selected_bhs_ordered:
        if bh not in soil_data_elev:
            continue
        x = positions_chainage[bh]
        top_elev = max(e[0] for e in soil_data_elev[bh])
        ax.text(x, top_elev + 3, bh, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(y_min, y_max)
    if positions_chainage:
        xmin = min(positions_chainage.values()) - half_width
        xmax = max(positions_chainage.values()) + 15*half_width
    else:
        xmin, xmax = -half_width, half_width
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Chainage along section (ft)", fontsize=16)
    ax.set_ylabel("Elevation (ft)", fontsize=16)
    ax.set_title(title, fontsize=18)

    legend_types = [s for s in ORDERED_SOIL_TYPES if s in used_types]
    legend_types += [s for s in sorted(unknown_types) if s not in legend_types]

    legend_patches = [
        mpatches.Patch(color=SOIL_COLOR_MAP.get(s, "#cccccc"), label=s)
        for s in legend_types
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    fig.tight_layout()
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
figw = st.sidebar.number_input("Figure width (in)", value=100.0, step=5.0)
figh = st.sidebar.number_input("Figure height (in)", value=20.0, step=1.0)

def make_base_map(center_latlon=(39.0, -104.9), zoom=13, basemap="Esri Satellite") -> folium.Map:
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
            folium.raster_layers.TileLayer(tiles=url, name=name, attr=attr[name], overlay=False, control=True, **{"time": "2012-01-01", "tilematrixset": "GoogleMapsCompatible", "maxZoom": 8}).add_to(fmap)
        else:
            folium.raster_layers.TileLayer(tiles=url, name=name, attr=attr[name], overlay=False, control=True).add_to(fmap)
    LayerControl(position='topright').add_to(fmap)
    return fmap

if main_file is None:
    st.info("Upload your MAIN borehole Excel to begin. Optional: also upload a PROPOSED.xlsx with lat/lon/name.")
    st.stop()

# Load MAIN
try:
    df = load_df_from_excel(main_file)
except Exception as e:
    st.error(f"Failed to read MAIN Excel: {e}")
    st.stop()

bh_coords = make_borehole_coords(df)

# Load PROPOSED (optional)
proposed_df = pd.DataFrame(columns=["Latitude","Longitude","Name"])
if prop_file is not None:
    # If you decide to use proposed points again, call load_proposed_df here.
    pass

# Center map
pts = [bh_coords[["Latitude","Longitude"]]]
all_pts = pd.concat(pts, ignore_index=True)
center_lat = float(all_pts['Latitude'].mean())
center_lon = float(all_pts['Longitude'].mean())

# Map and layers
fmap = make_base_map(center_latlon=(center_lat, center_lon), zoom=13, basemap=basemap_name)
for _, r in bh_coords.iterrows():
    add_labeled_point(fmap, float(r['Latitude']), float(r['Longitude']), str(r['Borehole']), EXISTING_TEXT_COLOR)

# Draw tool
draw = Draw(
    draw_options={
        "polyline": {"shapeOptions": {"color": "#3388ff", "weight": 4}},
        "polygon": False,
        "circle": False,
        "rectangle": False,
        "marker": False,
        "circlemarker": False,
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

# Pull a LineString from either 'last_active_drawing' or 'all_drawings'
def extract_linestring_from_map_out(mo) -> LineString | None:
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

# Persist the last valid section in session_state
if "section_line_coords" not in st.session_state:
    st.session_state["section_line_coords"] = None

maybe_line = extract_linestring_from_map_out(map_out or {})
if maybe_line is not None:
    st.session_state["section_line_coords"] = list(map(list, maybe_line.coords))

# Controls
st.subheader("Generate Soil Profile")
go = st.button("Generate Soil Profile", type="primary")

if not st.session_state["section_line_coords"]:
    st.info("Draw a polyline on the map, then click **Generate Soil Profile**.")
    st.stop()

# Rebuild LineString from session_state when button is pressed
last_line = LineString(st.session_state["section_line_coords"])

if not go:
    st.info("Click **Generate Soil Profile** to compute the corridor, table, and plot.")
    st.stop()

# Compute corridor selection
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

report = chainage_and_offset_df(bh_coords, last_line)
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

# Plot
try:
    fig = plot_soil_profile_original_style(
        df=df[df['Borehole'].isin(selected_bhs_ordered)],
        selected_bhs_ordered=selected_bhs_ordered,
        positions_chainage=positions_chainage,
        y_min=float(ymin),
        y_max=float(ymax),
        title=title,
        figsize=(float(figw), float(figh))
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.getvalue()
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.error(f"Plot failed: {e}")
    st.stop()

# Downloads
csv_buf = io.StringIO()
sel.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode("utf-8")

gj = {
    "type": "Feature",
    "properties": {"name": "section_line"},
    "geometry": {"type": "LineString", "coordinates": st.session_state['section_line_coords']}
}
gj_bytes = json.dumps(gj, indent=2).encode("utf-8")

st.download_button("Download CSV (section_boreholes.csv)", data=csv_bytes, file_name="section_boreholes.csv", mime="text/csv", use_container_width=True)
st.download_button("Download section line (section_line.geojson)", data=gj_bytes, file_name="section_line.geojson", mime="application/geo+json", use_container_width=True)
st.download_button("Download plot (soil_profile.png)", data=png_bytes, file_name="soil_profile.png", mime="image/png", use_container_width=True)

st.caption("✅ Draw your section, click **Generate Soil Profile**, then download the CSV/GeoJSON/PNG.")
