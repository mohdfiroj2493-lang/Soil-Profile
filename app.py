# app.py — Borehole Mapping and Section Profiling Tool
# ✨ Fully updated
#   • Multi-sheet support for EXISTING and PROPOSED Excel files
#   • Distinct colors per sheet
#   • Sidebar toggles + color legend
#   • Integrated 2D and 3D profile plotting
#   • High-quality figure export options

from typing import Dict, List, Tuple, Optional
import io, math
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import Map, LayerControl
from folium.plugins import Draw
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ── Visual config ────────────────────────────────────────────────────────────
EXISTING_TEXT_COLORS = [
    "#1e88e5", "#43a047", "#8e24aa", "#fb8c00", "#6d4c41",
    "#3949ab", "#00897b", "#fdd835"
]
PROPOSED_TEXT_COLORS = [
    "#e53935", "#8e24aa", "#3949ab", "#00897b", "#fdd835",
    "#6d4c41", "#43a047", "#fb8c00"
]
CIRCLE_RADIUS_PX, CIRCLE_STROKE, CIRCLE_STROKE_W = 10, "#ffffff", 1
CIRCLE_FILL_OPACITY, LABEL_DX_PX, LABEL_DY_PX = 0.95, 8, -10
TITLE_DEFAULT = "Soil Profile"
FIG_HEIGHT_IN = 22.0

# ── Soil color map and renaming rules ────────────────────────────────────────
SOIL_COLOR_MAP = {
    "Topsoil": "#ffffcb", "Water": "#00ffff",
    # Clays/Silts
    "CL": "#c5cae9","CH": "#64b5f6","CL-CH": "#fff176","CL-ML": "#ef9a9a",
    "ML": "#ef5350","MH": "#ffb74d","ML-CL": "#dbb40c",
    # Gravels/Sands
    "GM": "#aed581","GW": "#c8e6c9","GC": "#00ff00","GP": "#aaff32","GP-GC": "#008000","GP-GM": "#15b01a",
    "SM": "#76d7c4","SP": "#ce93d8","SC": "#81c784","SW": "#ba68c8",
    "SM-SC": "#e1bee7","SM-ML": "#dcedc8","SC-CL": "#ffee58","SC-SM": "#fff59d","SP-SM": "#c1f80a","SP-SC": "#7fff00","SW-SM": "#90ee90",
    # Rock / Fill / Weathered
    "PWR": "#808080","RF": "#929591","Rock": "#c0c0c0", "BASALT CONGLOMERATE": "#ffff14","BASALT": "#bbf90f","BRECCIA": "#f5deb3","CHERT": "#a0522d",
    "CLAYSTONE": "#40e0d0","DIATOMITE": "#7bc8f6","DOLOMITE": "#000080", "LIMESTONE": "#006400","MUDSTONE": "#add8e6","SANDSTONE": "#c79fef","SHALE": "#13eac9",
    "SILTSTONE": "#ffd700",
}
ORDERED_SOIL_TYPES = [
    "Topsoil", "Water",
    "SM", "SM-ML", "SM-SC", "SP-SM", "SP-SC", "SP", "SW",
    "SC", "SC-CL", "SC-SM", "SP-SC","SW-SM",
    "CL", "CL-ML", "CL-CH", "CH",
    "ML", "MH", "ML-CL", 
    "GM", "GP-GM", "GP-GC", "GP", "GC", "GW",
    "Rock", "PWR", "RF", "BASALT CONGLOMERATE", "BASALT", "BRECCIA", "CHERT",
    "CLAYSTONE", "DIATOMITE", "DOLOMITE", "LIMESTONE", "MUDSTONE", "SANDSTONE", "SHALE",
    "SILTSTONE",
]
RENAME_MAP = {
    "Bore Log": "Borehole", "Elevation From": "Elevation_From",
    "Elevation To": "Elevation_To",
    "Soil Layer Description": "Soil_Type",
    "Latitude": "Latitude", "Longitude": "Longitude",
    "SPT N-Value": "SPT", "Elevation Water Table": "Water_Elev",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_spt_avg(value):
    """Return ONLY the numeric average SPT value (e.g., 9.2) or '' if not valid."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.upper() == "N/A" or s == "":
        return ""
    nums = []
    for x in s.split(","):
        try:
            nums.append(float(x.strip().replace('"', '')))
        except ValueError:
            pass
    return round(sum(nums) / len(nums), 2) if nums else ""

@st.cache_data(show_spinner=False)
def load_multisheet_existing(uploaded_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Load multi-sheet EXISTING borehole Excel.
    Each sheet may represent a different dataset — up to 8 sheets supported.
    Returns a dict {sheet_name: DataFrame}.
    """
    all_sheets = pd.read_excel(io.BytesIO(uploaded_bytes), sheet_name=None)
    result: Dict[str, pd.DataFrame] = {}

    for i, (sheet, df) in enumerate(all_sheets.items()):
        # Skip empty sheets
        if df.empty:
            continue

        # Clean headers and rename to unified names
        df.columns = df.columns.str.strip()
        df.rename(columns=RENAME_MAP, inplace=True, errors="ignore")

        # Ensure Latitude/Longitude columns exist and are numeric
        if not {"Latitude", "Longitude"}.issubset(df.columns):
            continue
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df = df.dropna(subset=["Latitude", "Longitude"]).copy()

        # Compute average SPT labels (N-values)
        df["SPT_Label"] = df.get("SPT", pd.NA).apply(compute_spt_avg)

        # 🟡 Normalize soil names: extract codes in parentheses or detect Topsoil
        soil_col = None
        for c in df.columns:
            if str(c).strip().lower() in ["soil_type", "soil layer description", "soil layer"]:
                soil_col = c
                break
        if soil_col:
            df[soil_col] = df[soil_col].astype(str)
            df[soil_col] = df[soil_col].str.extract(r"\((.*?)\)").fillna(
                df[soil_col].str.replace(r"^.*top\s*soil.*$", "Topsoil", case=False, regex=True)
            )
            df.rename(columns={soil_col: "Soil_Type"}, inplace=True)
        # Assign metadata
        df["Sheet"] = sheet
        df["Color"] = EXISTING_TEXT_COLORS[i % len(EXISTING_TEXT_COLORS)]

        result[sheet] = df

    return result

@st.cache_data(show_spinner=False)
def normalize_cols_general(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    if "latitude" in lower: ren[lower["latitude"]] = "Latitude"
    elif "lat" in lower: ren[lower["lat"]] = "Latitude"
    if "longitude" in lower: ren[lower["longitude"]] = "Longitude"
    elif "lon" in lower: ren[lower["lon"]] = "Longitude"
    elif "long" in lower: ren[lower["long"]] = "Longitude"
    if "name" in lower: ren[lower["name"]] = "Name"
    elif "id" in lower: ren[lower["id"]] = "Name"
    df = df.rename(columns=ren)
    if "Name" not in df.columns:
        df["Name"] = [f"Proposed-{i+1}" for i in range(len(df))]
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df.dropna(subset=["Latitude","Longitude"])[["Latitude","Longitude","Name"]]

@st.cache_data(show_spinner=False)
def load_proposed_multisheet(uploaded_bytes: bytes) -> Dict[str, pd.DataFrame]:
    all_sheets = pd.read_excel(io.BytesIO(uploaded_bytes), sheet_name=None)
    result = {}
    for i, (sheet, df) in enumerate(all_sheets.items()):
        if i >= 8: break
        df_norm = normalize_cols_general(df)
        if not df_norm.empty:
            df_norm["Sheet"] = sheet
            df_norm["Color"] = PROPOSED_TEXT_COLORS[i % len(PROPOSED_TEXT_COLORS)]
            result[sheet] = df_norm
    return result

# ── Geometry helpers ────────────────────────────────────────────────────────
def chainage_and_offset_ft(line: LineString, lat: float, lon: float):
    proj = line.project(Point(lon, lat))
    frac = proj / line.length if line.length > 0 else 0.0
    total_len = 0.0
    coords = list(line.coords)
    for i in range(1, len(coords)):
        total_len += geodesic(
            (coords[i - 1][1], coords[i - 1][0]),
            (coords[i][1], coords[i][0])
        ).feet
    chain = total_len * frac
    nearest = line.interpolate(proj)
    off = geodesic((lat, lon), (nearest.y, nearest.x)).feet
    return float(chain), float(off)

def latlon_to_local_xy_ft(lat, lon, lat0, lon0):
    x_m = geodesic((lat0, lon0), (lat0, lon)).meters
    y_m = geodesic((lat0, lon0), (lat, lon0)).meters
    x = x_m * 3.28084 * (1 if lon >= lon0 else -1)
    y = y_m * 3.28084 * (1 if lat >= lat0 else -1)
    return x, y

def auto_y_limits(df, pad_ratio=0.05):
    if df.empty:
        return 0, 1
    y_min = float(df['Elevation_To'].min())
    y_max = float(df['Elevation_From'].max())
    rng = max(1.0, (y_max - y_min))
    pad = rng * pad_ratio
    return y_min - pad, y_max + pad

# ── Map helpers ──────────────────────────────────────────────────────────────
def add_labeled_point(fmap, lat, lon, name, color_hex):
    folium.CircleMarker(
        location=(lat, lon),
        radius=CIRCLE_RADIUS_PX,
        color=CIRCLE_STROKE,
        weight=CIRCLE_STROKE_W,
        fill=True,
        fill_color=color_hex,
        fill_opacity=CIRCLE_FILL_OPACITY
    ).add_to(fmap)

    label_html = (
        f"<div style='background:transparent;border:none;box-shadow:none;pointer-events:none;"
        f"padding:0;margin:0;transform: translate({LABEL_DX_PX}px, {LABEL_DY_PX}px);"
        f"display:inline-block;white-space:nowrap;font-size:13px;font-weight:700;color:{color_hex};"
        f"text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;'>{name}</div>"
    )

    folium.Marker(
        location=(lat, lon),
        icon=folium.DivIcon(html=label_html, icon_size=(0, 0))
    ).add_to(fmap)

# ── UI Setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Borehole Section Profile", layout="wide")
st.sidebar.header("Upload Files")

main_file = st.sidebar.file_uploader("MAIN borehole Excel (multi-sheet OK)", type=["xlsx", "xls"])
prop_file = st.sidebar.file_uploader("Optional PROPOSED.xlsx (multi-sheet OK)", type=["xlsx", "xls"])

if main_file is None:
    st.title("Map with Bore Logs")
    st.info("Upload the MAIN Excel to begin.")
    st.stop()

# ── Load multi-sheet EXISTING boreholes ─────────────────────────────────────
existing_dict = load_multisheet_existing(main_file.getvalue())
if not existing_dict:
    st.error("No valid borehole sheets found in uploaded file.")
    st.stop()

# Sidebar toggles + color legend
st.sidebar.markdown("### Existing Sheets")
selected_sheets = []
for sheet, df_sheet in existing_dict.items():
    color = df_sheet["Color"].iloc[0]
    chk = st.sidebar.checkbox(f"🟢 {sheet}", value=True, help=f"Color: {color}", key=f"chk_{sheet}")
    if chk:
        selected_sheets.append(sheet)
st.sidebar.markdown("---")

# Combine selected sheets into single DataFrame
df = pd.concat([existing_dict[s] for s in selected_sheets if not existing_dict[s].empty], ignore_index=True)

# ── Load optional PROPOSED points ───────────────────────────────────────────
proposed_dict = {}
if prop_file is not None:
    try:
        proposed_dict = load_proposed_multisheet(prop_file.getvalue())
    except Exception as e:
        st.warning(f"Could not read Proposed workbook: {e}")
        proposed_dict = {}

# ── Map Display ─────────────────────────────────────────────────────────────
st.title("Map with Bore Logs")

center_lat = float(df["Latitude"].mean())
center_lon = float(df["Longitude"].mean())

fmap = Map(location=(center_lat, center_lon), zoom_start=13, control_scale=True)
folium.raster_layers.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    name="Esri Satellite", attr="Tiles © Esri", overlay=False, control=True
).add_to(fmap)

# Add EXISTING boreholes grouped by sheet
for sheet, dfx in existing_dict.items():
    if sheet not in selected_sheets or dfx.empty:
        continue
    fg = folium.FeatureGroup(name=f"Existing — {sheet}", show=True)
    color = dfx["Color"].iloc[0]
    for _, r in dfx.iterrows():
        nm = str(r.get("Borehole", "") or f"{sheet}-BH")
        add_labeled_point(fg, float(r["Latitude"]), float(r["Longitude"]), nm, color)
    fg.add_to(fmap)

# Add PROPOSED boreholes grouped by sheet
for sheet, dfp in proposed_dict.items():
    if dfp.empty:
        continue
    fg = folium.FeatureGroup(name=f"Proposed — {sheet}", show=True)
    color = dfp["Color"].iloc[0]
    for _, r in dfp.iterrows():
        nm = str(r.get("Name", "")) or "Proposed"
        add_labeled_point(fg, float(r["Latitude"]), float(r["Longitude"]), nm, color)
    fg.add_to(fmap)

# Drawing tools and layer toggle
Draw(
    draw_options={"polyline": {"shapeOptions": {"color": "#0000ff", "weight": 6}},
                  "polygon": False, "circle": False, "rectangle": False,
                  "marker": False, "circlemarker": False},
    edit_options={"edit": True, "remove": True}
).add_to(fmap)
LayerControl(position="topright").add_to(fmap)

map_out = st_folium(fmap, height=600, use_container_width=True,
                    returned_objects=["last_active_drawing", "all_drawings"], key="map")
# ── Extract drawn section line ──────────────────────────────────────────────
def extract_linestring(mo):
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

# ── Section / Profile generation ────────────────────────────────────────────
st.title("Section / Profile (ft) — Soil")
corridor_ft = st.slider("Corridor width (ft)", 0, 1000, 200, 10)

colA, colB = st.columns([1, 1])
with colA:
    show_codes = st.checkbox("Show soil code (ML/SM/...)", value=False)
with colB:
    show_spt = st.checkbox("Show SPT N value", value=True)

if not st.session_state["section_line_coords"]:
    st.info("Draw a polyline on the map (double-click to finish). The profiles will appear below automatically.")
    st.stop()

section_line = LineString(st.session_state["section_line_coords"])

# Select EXISTING boreholes near the section corridor
bh_coords = (
    df.groupby("Borehole")[["Latitude", "Longitude"]].first().reset_index()
    if "Borehole" in df.columns else
    df.rename(columns={"Name": "Borehole"})  # fallback if Borehole col missing
)
rows = []
for _, r in bh_coords.iterrows():
    ch, off = chainage_and_offset_ft(section_line, float(r["Latitude"]), float(r["Longitude"]))
    if off <= float(corridor_ft):
        rows.append({"Borehole": str(r.get("Borehole", "")), "Chainage_ft": round(ch, 2)})

if not rows:
    st.warning(f"No EXISTING boreholes within {corridor_ft:.0f} ft of the drawn section.")
    st.stop()

ordered_bhs = [r["Borehole"] for r in sorted(rows, key=lambda x: (x["Chainage_ft"], x["Borehole"]))]
xpos = {r["Borehole"]: r["Chainage_ft"] for r in rows}

# ── 2D & 3D plotting utils (width & grid step) ─────────────────────────────
def dynamic_column_width(x_positions, default_width=60.0, min_width=8.0, fraction_of_min_gap=0.8):
    xs = sorted(x_positions.values())
    if len(xs) < 2:
        return default_width
    gaps = [xs[i+1] - xs[i] for i in range(len(xs) - 1) if xs[i+1] > xs[i]]
    if not gaps:
        return default_width
    min_gap = min(gaps)
    width = min(default_width, fraction_of_min_gap * min_gap)
    return max(min_width, width)

def _nice_step(rng: float, target: int = 10) -> float:
    """Choose a 'nice' step size for grid spacing over range rng."""
    if rng <= 0:
        return 1.0
    rough = rng / max(1, target)
    expv = math.floor(math.log10(rough))
    frac = rough / (10 ** expv)
    if frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 2.5:
        nice = 2.5
    elif frac <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** expv)

# ── 2D Profile Builder (Plotly) ─────────────────────────────────────────────
def build_plotly_profile(
    df: pd.DataFrame, ordered_bhs: List[str], x_positions: Dict[str, float],
    y_min: float, y_max: float, title: str,
    column_width: Optional[float],
    show_codes: bool = False, show_spt: bool = True,
    fig_height_px: int = 1000,
) -> go.Figure:
    # Width is either manual (passed) or auto from spacing
    width = column_width if column_width is not None else dynamic_column_width(x_positions)
    half = width / 2.0

    xs = list(x_positions.values())
    xmin = (min(xs) - half) if xs else -half
    xmax = (max(xs) + 3 * half) if xs else half

    # Adaptive label sizes depending on width
    def inner_font_for(w):
        if w >= 60: return 12
        if w >= 45: return 11
        if w >= 30: return 10
        if w >= 18: return 9
        return 8

    def bh_font_for(w):
        if w >= 60: return 14
        if w >= 45: return 13
        if w >= 30: return 12
        if w >= 18: return 11
        return 10

    inner_font = inner_font_for(width)
    bh_font = bh_font_for(width)
    label_font = max(inner_font + 2, 12)

    grid_lines: List[dict] = []
    soil_rects: List[dict] = []
    annotations = []
    used_types = set()

    # Background grid (behind rectangles)
    yrng = max(1.0, y_max - y_min)
    xrng = max(1.0, xmax - xmin)
    y_step = _nice_step(yrng, target=50)
    x_step = _nice_step(xrng, target=12)

    # Horizontal lines
    y0 = math.floor(y_min / y_step) * y_step
    y1 = math.ceil(y_max / y_step) * y_step
    yv = y0
    while yv <= y1 + 1e-9:
        grid_lines.append(dict(
            type="line", x0=xmin, x1=xmax, y0=yv, y1=yv,
            line=dict(color="#eaeaea", width=1),
            layer="below", xref="x", yref="y"
        ))
        yv += y_step

    # Vertical lines
    x0 = math.floor(xmin / x_step) * x_step
    x1 = math.ceil(xmax / x_step) * x_step
    xv = x0
    while xv <= x1 + 1e-9:
        grid_lines.append(dict(
            type="line", x0=xv, x1=xv, y0=y_min, y1=y_max,
            line=dict(color="#f0f0f0", width=1),
            layer="below", xref="x", yref="y"
        ))
        xv += x_step

    # Collect water elevations per BH (one marker per BH)
    water_x, water_y = [], []

    # Soil rectangles + labels
    for bh in ordered_bhs:
        bore = df[df["Borehole"] == bh]
        if bore.empty:
            continue
        x = x_positions[bh]
        top_el = bore["Elevation_From"].max()
        annotations.append(dict(
            x=x, y=top_el + 3, text=bh, showarrow=False,
            xanchor="center", yanchor="bottom",
            font=dict(size=bh_font, family="Arial Black", color="#111")
        ))

        if "Water_Elev" in bore.columns:
            wt_series = pd.to_numeric(bore["Water_Elev"], errors="coerce").dropna()
            if not wt_series.empty:
                try:
                    wt = float(wt_series.iloc[0])
                    water_x.append(x)
                    water_y.append(wt)
                except (ValueError, TypeError):
                    pass

        for _, r in bore.iterrows():
            ef, et = float(r["Elevation_From"]), float(r["Elevation_To"])
            soil = str(r["Soil_Type"])
            spt_val = r.get("SPT_Label", "")
            spt = "" if pd.isna(spt_val) else str(spt_val)

            color = SOIL_COLOR_MAP.get(soil, "#cccccc")
            used_types.add(soil)

            # Rectangle for the soil layer
            soil_rects.append(dict(
                type="rect", x0=x - half, x1=x + half, y0=et, y1=ef,
                line=dict(color="#000", width=1.3),
                fillcolor=color, layer="below",
            ))

            mid_y = (ef + et) / 2.0
            offset = max(4.0, half * 0.15)  # small horizontal offset from rectangle edge

            # Soil code OUTSIDE on the LEFT
            if show_codes:
                annotations.append(dict(
                    x=x - half - offset,
                    y=mid_y,
                    text=soil,
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    font=dict(size=label_font, family="Arial", color="#111"),
                ))

            # SPT value OUTSIDE on the RIGHT (just the number)
            if show_spt and spt not in ("", "nan"):
                annotations.append(dict(
                    x=x + half + offset,
                    y=mid_y,
                    text=spt,
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=label_font, family="Arial", color="#111"),
                ))

    # Legend (preferred order first, then extras)
    ordered_present = [s for s in ORDERED_SOIL_TYPES if s in used_types]
    extra_present = sorted([s for s in used_types if s not in set(ORDERED_SOIL_TYPES)])
    legend_types = ordered_present + extra_present

    fig = go.Figure()
    for soil in legend_types:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=SOIL_COLOR_MAP.get(soil, "#cccccc")),
            name=soil, showlegend=True
        ))

    if water_x:
        fig.add_trace(go.Scatter(
            x=water_x, y=water_y, mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="#1e88e5"),
            name="Water Table",
            hovertemplate="Water Elev: %{y:.2f} ft<extra></extra>"
        ))

        fig.update_layout(
            title=dict(text=title,font=dict(color="black", size=18),),
            font=dict(family="Inter, Arial, sans-serif"),xaxis=dict(title=dict(text="Chainage along section (ft)",font=dict(color="black", size=14),)),yaxis=dict(title=dict(text="Elevation (ft)",font=dict(color="black", size=14),) ),
            shapes=grid_lines + soil_rects,
            annotations=annotations,
            height=fig_height_px,
            margin=dict(l=70, r=260, t=70, b=70),
            plot_bgcolor="white", legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, bordercolor="#ddd",borderwidth=1,), )
    fig.update_xaxes(range=[xmin, xmax], tickfont=dict(color="black", size=12), showgrid=False, zeroline=False)
    fig.update_yaxes(range=[y_min, y_max], tickfont=dict(color="black", size=12), showgrid=False, zeroline=False)
    return fig

# ── Generate 2D profile ─────────────────────────────────────────────────────
plot_df = df[df["Borehole"].isin(ordered_bhs)]
ymin_auto, ymax_auto = auto_y_limits(plot_df)
fig_height_px = int(FIG_HEIGHT_IN * 50)

suggested = dynamic_column_width(xpos)
auto_width = st.checkbox("Auto column width", value=True)
if auto_width:
    column_width_ft = None
    st.caption(f"Auto width ≈ **{suggested:.1f} ft** (80% of nearest spacing)")
else:
    minw, maxw = 4.0, 300.0
    default_val = float(min(max(suggested, 30.0), maxw))
    column_width_ft = st.slider("Column width (ft)", minw, maxw, default_val, 2.0)

fig2d = build_plotly_profile(
    df=plot_df, ordered_bhs=ordered_bhs, x_positions=xpos,
    y_min=ymin_auto, y_max=ymax_auto, title=TITLE_DEFAULT,
    column_width=column_width_ft, show_codes=show_codes, show_spt=show_spt,
    fig_height_px=fig_height_px
)
st.plotly_chart(
    fig2d, use_container_width=True,
    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "soil_profile", "scale": 4}}
)

st.markdown("### Pattern-Filled Soil Profile (Static)")
fig_mat = plot_profile_matplotlib(plot_df, ordered_bhs, xpos, ymin_auto, ymax_auto)
st.pyplot(fig_mat)
SOIL_HATCH_MAP = {
    "SM": "xx",
    "SC": "///",
    "SP": "...",
    "CL": "\\\\",
    "CH": "++",
    "GM": "oo",
    "GP-GM":"oo",
    "GP-GC":"oo",
    "GC": "xx",
    "MH": "...",
    "Topsoil": "////",
    "Rock": "////",
}
def plot_profile_matplotlib(df, ordered_bhs, x_positions, y_min, y_max, col_width=8.0):
    fig, ax = plt.subplots(figsize=(14, 8))
    half = col_width / 2.0
    for bh in ordered_bhs:
        bore = df[df["Borehole"] == bh]
        if bore.empty:
            continue
        x = x_positions[bh]
        # Draw soil layers with hatching
        for _, r in bore.iterrows():
            ef, et = float(r["Elevation_From"]), float(r["Elevation_To"])
            soil = str(r["Soil_Type"])
            hatch = SOIL_HATCH_MAP.get(soil, "")
            rect = plt.Rectangle(
                (x - half, et),
                col_width,
                ef - et,
                edgecolor="black",
                facecolor="none",
                hatch=hatch,
                linewidth=1.2
            )
            ax.add_patch(rect)
        # Borehole label at top
        top_el = bore["Elevation_From"].max()
        ax.text(x, top_el + 2, bh, ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xlim(min(x_positions.values()) - col_width*2,
                max(x_positions.values()) + col_width*2)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.set_xlabel("Chainage along section (ft)")
    ax.set_ylabel("Elevation (ft)")
    ax.grid(True, axis="y", color="#bbbbbb", linewidth=0.6, alpha=0.6)
    return fig
# ── 3D profile (PLAN COORDS) builder ────────────────────────────────────────
def build_3d_profile_plan(
    df: pd.DataFrame,
    selected_bhs: List[str],
    xy_ft: Dict[str, Tuple[float, float]],
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

    # Legend ordering similar to 2D
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

# ── 3D Borehole View (PLAN) ─────────────────────────────────────────────────
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
xy_map = {
    row['Borehole']: latlon_to_local_xy_ft(float(row['Latitude']), float(row['Longitude']), lat0, lon0)
    for _, row in sel_coords.iterrows()
}

plot_df3d = df[df['Borehole'].isin(bhs_for_3d)]
ymin_auto3d, ymax_auto3d = auto_y_limits(plot_df3d)

fig3d = build_3d_profile_plan(
    df=plot_df3d, selected_bhs=bhs_for_3d, xy_ft=xy_map,
    y_min=ymin_auto3d, y_max=ymax_auto3d, title=TITLE_DEFAULT,
    column_width_ft=60.0, vert_exag=vert_exag
)
st.plotly_chart(
    fig3d, use_container_width=True,
    config={"displaylogo": False, "toImageButtonOptions": {"scale": 3}}
)
