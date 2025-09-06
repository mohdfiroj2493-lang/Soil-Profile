# app.py
import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# Constants / helpers
# -----------------------------------------------------------------------------
FT_PER_M = 3.280839895
R_EARTH_M = 6371008.8

def pick(cols, *aliases):
    """Return first matching column (lowercased) from possible aliases, else None."""
    s = {c.lower(): c for c in cols}
    for a in aliases:
        a = a.lower()
        if a in s:
            return s[a]
    return None

def to_local_xy_m(lat, lon, lat0, lon0):
    """
    Convert (lat,lon) to local planar X,Y in meters using a simple local-Mercator
    approximation anchored at (lat0, lon0). Good for corridor-length work.
    """
    latr = np.radians(lat)
    lonr = np.radians(lon)
    lat0r = math.radians(lat0)
    lon0r = math.radians(lon0)
    x = R_EARTH_M * (lonr - lon0r) * math.cos(lat0r)
    y = R_EARTH_M * (latr - lat0r)
    return x, y

def project_chainage_to_polyline(xy_pts_m, xy_line_m):
    """Project points onto a polyline (all in meters). Return chain_ft, dist_ft, total_len_ft."""
    if len(xy_line_m) < 2:
        n = len(xy_pts_m)
        return np.zeros(n), np.full(n, np.inf), 0.0

    seg_vecs = xy_line_m[1:] - xy_line_m[:-1]
    seg_len = np.linalg.norm(seg_vecs, axis=1)
    seg_len[seg_len == 0] = 1e-12
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    chain = np.zeros(len(xy_pts_m))
    dist = np.full(len(xy_pts_m), np.inf)

    for i, P in enumerate(xy_pts_m):
        best_d = np.inf
        best_ch = 0.0
        for k, v in enumerate(seg_vecs):
            A = xy_line_m[k]
            L2 = np.dot(v, v)
            t = np.clip(np.dot(P - A, v) / L2, 0.0, 1.0)
            Q = A + t * v
            d = np.linalg.norm(P - Q)
            if d < best_d:
                best_d   = d
                best_ch  = cum[k] + t * seg_len[k]
        chain[i] = best_ch
        dist[i]  = best_d

    return chain * FT_PER_M, dist * FT_PER_M, cum[-1] * FT_PER_M

def first_nonnull_string(series):
    for v in series:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def avg_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return float(s.mean())
    return None

def soil_palette():
    # categorical palette up to ~12 types; repeats if needed
    return [
        "#10b981","#f97316","#6366f1","#e11d48","#14b8a6","#f59e0b",
        "#8b5cf6","#ef4444","#0ea5e9","#84cc16","#a855f7","#f43f5e"
    ]

def color_for_soil(code, mapping, palette):
    if code not in mapping:
        mapping[code] = palette[len(mapping) % len(palette)]
    return mapping[code]

def same_polyline(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

# -----------------------------------------------------------------------------
# Page setup & session state
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Borehole / Soil Profile", layout="wide")
st.title("🧱 Borehole Map & Soil Profile")

if "draw_counter" not in st.session_state:
    st.session_state["draw_counter"] = 0
if "section_line" not in st.session_state:
    st.session_state["section_line"] = None  # GeoJSON-like [[lon,lat],...]

# -----------------------------------------------------------------------------
# Uploaders
# -----------------------------------------------------------------------------
st.sidebar.header("Upload")
main_file = st.sidebar.file_uploader("Upload MAIN Excel (required)", type=["xls", "xlsx"])
prop_file = st.sidebar.file_uploader("Upload PROPOSED Excel (optional: lat/lon/name)", type=["xls", "xlsx"])

# -----------------------------------------------------------------------------
# Load MAIN
# -----------------------------------------------------------------------------
if not main_file:
    st.info("Upload your **MAIN** bore log Excel to begin.")
    st.stop()

raw = pd.read_excel(main_file)
raw.columns = [c.strip() for c in raw.columns]

# Flexible column aliases
c_name   = pick(raw.columns, "Name","ID","Boring ID","Hole ID","Borehole","BH")
c_lat    = pick(raw.columns, "Latitude","Lat","LAT")
c_lon    = pick(raw.columns, "Longitude","Lon","Long","LON")
c_top    = pick(raw.columns, "Top EL","Ground Elevation","Boring Elevation","Ground EL","Top Elevation")
c_depth  = pick(raw.columns, "Depth","Total Depth","Hole Depth")
c_bottom = pick(raw.columns, "Bottom EL","Bottom Elevation")
c_soil   = pick(raw.columns, "USCS","Soil","Soil Type","Classification","USCS Group")
c_n      = pick(raw.columns, "N","SPT","SPT N","N-Value","N value","SPT N-value")

# Build "point-level" table (may contain multiple rows per bore)
df = pd.DataFrame({
    "Name":      raw[c_name] if c_name else np.arange(len(raw)).astype(str),
    "Latitude":  pd.to_numeric(raw[c_lat], errors="coerce") if c_lat else np.nan,
    "Longitude": pd.to_numeric(raw[c_lon], errors="coerce") if c_lon else np.nan,
    "Soil":      raw[c_soil] if c_soil else None,
    "N":         pd.to_numeric(raw[c_n], errors="coerce") if c_n else np.nan,
})

# Elevations
top_el = pd.to_numeric(raw[c_top], errors="coerce") if c_top else np.nan
if c_bottom:
    bottom_el = pd.to_numeric(raw[c_bottom], errors="coerce")
elif c_depth:
    depth = pd.to_numeric(raw[c_depth], errors="coerce")
    bottom_el = top_el - depth
else:
    bottom_el = np.nan

df["Top_EL"] = top_el
df["Bottom_EL"] = bottom_el

# Drop rows with missing coordinates
df = df.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
if df.empty:
    st.error("No valid rows with Latitude/Longitude found in MAIN Excel.")
    st.stop()

# Per-boring aggregates (for profile & labels)
agg = (
    df.groupby("Name")
      .agg({
          "Latitude":"first",
          "Longitude":"first",
          "Top_EL":"first",
          "Bottom_EL":"first",
          "Soil": first_nonnull_string,
          "N":   avg_numeric
      })
      .reset_index()
)
agg["N_text"] = agg["N"].apply(lambda v: None if pd.isna(v) else f"{int(round(v))}")
agg["Soil"] = agg["Soil"].fillna("Unknown")

# -----------------------------------------------------------------------------
# Optional PROPOSED
# -----------------------------------------------------------------------------
prop = None
if prop_file:
    rp = pd.read_excel(prop_file)
    rp.columns = [c.strip() for c in rp.columns]
    p_name = pick(rp.columns, "Name","ID","Borehole","BH","Label")
    p_lat  = pick(rp.columns, "Latitude","Lat")
    p_lon  = pick(rp.columns, "Longitude","Lon","Long")
    if p_lat and p_lon:
        prop = pd.DataFrame({
            "Name":      rp[p_name] if p_name else "PROPOSED",
            "Latitude":  pd.to_numeric(rp[p_lat], errors="coerce"),
            "Longitude": pd.to_numeric(rp[p_lon], errors="coerce"),
        }).dropna(subset=["Latitude","Longitude"])
        if prop.empty:
            prop = None

# -----------------------------------------------------------------------------
# Map with draw tool
# -----------------------------------------------------------------------------
st.subheader("Map — draw your section")
st.caption("Click the polyline tool, add vertices, and **double-click** to finish. "
           "Drawing a new line **replaces** the previous one. Use the small 'Finish' button if needed.")

center_lat, center_lon = float(agg["Latitude"].mean()), float(agg["Longitude"].mean())
m = folium.Map(location=[center_lat, center_lon], zoom_start=14, control_scale=True)

# Base layer
folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)

# MAIN markers + labels (one line, no wrap)
for _, r in agg.iterrows():
    popup = folium.Popup(
        f"<b>{r['Name']}</b><br>"
        f"Soil: {r['Soil']}<br>"
        f"N̄: {r['N_text'] if r['N_text'] else 'N/A'}<br>"
        f"Top EL: {r['Top_EL']:.2f} ft<br>"
        f"Bottom EL: {r['Bottom_EL']:.2f} ft",
        max_width=350
    )
    folium.CircleMarker(
        [r["Latitude"], r["Longitude"]],
        radius=6, color="#2563eb", fill=True, fill_color="#2563eb", fill_opacity=0.8, popup=popup
    ).add_to(m)
    folium.Marker(
        [r["Latitude"], r["Longitude"]],
        icon=folium.DivIcon(
            html=f"""
            <div style="font-size: 12px; color: black; white-space: nowrap; 
                        background: rgba(255,255,255,.6); padding: 0 2px; border-radius: 3px;
                        position: relative; top: 10px;">
                {str(r['Name'])}
            </div>
            """
        )
    ).add_to(m)

# PROPOSED markers (if any)
if prop is not None:
    for _, r in prop.iterrows():
        folium.CircleMarker(
            [r["Latitude"], r["Longitude"]],
            radius=6, color="#ea580c", fill=True, fill_color="#ea580c", fill_opacity=0.9,
            popup=folium.Popup(f"<b>Proposed:</b> {r['Name']}", max_width=300)
        ).add_to(m)

# Draw control
Draw(
    export=False,
    position="topleft",
    draw_options={
        "polyline": {"shapeOptions": {"color": "#ef4444","weight": 4}},
        "polygon": False, "rectangle": False, "circle": False,
        "circlemarker": False, "marker": False,
    },
    edit_options={"edit": True, "remove": True},
).add_to(m)

folium.LayerControl().add_to(m)

# Keep only the latest drawn line visually by re-keying the widget after change
map_key = f"map_{st.session_state['draw_counter']}"
map_state = st_folium(m, height=520, use_container_width=True,
                      returned_objects=["last_active_drawing","all_drawings"],
                      key=map_key)

# Extract the last drawn polyline from map_state
latest_line = None
drawings = map_state.get("all_drawings") or []
for g in reversed(drawings):
    if g and g.get("type") == "Feature":
        geom = g.get("geometry", {})
        if geom.get("type") == "LineString":
            latest_line = geom.get("coordinates")  # [[lon,lat],...]
            break

# Detect change and force a visual reset so the old line disappears
if latest_line is not None and not same_polyline(latest_line, st.session_state["section_line"]):
    st.session_state["section_line"] = latest_line
    st.session_state["draw_counter"] += 1
    st.rerun()

# Clear button
cols_top = st.columns([1,1,6])
with cols_top[0]:
    if st.button("🧹 Clear section"):
        st.session_state["section_line"] = None
        st.session_state["draw_counter"] += 1
        st.rerun()

# -----------------------------------------------------------------------------
# Section / Profile (disconnected soil columns)
# -----------------------------------------------------------------------------
st.subheader("Section / Profile (ft)")
corridor = st.slider("Corridor width (ft)", 25, 1000, 200, step=25)

if st.session_state["section_line"] is None:
    st.info("Draw a polyline on the map to compute chainage and show the profile.")
    st.stop()

# Convert to local XY
lat0, lon0 = center_lat, center_lon
line_xy_m = np.array([to_local_xy_m(lat=lat, lon=lon, lat0=lat0, lon0=lon0)[::-1][::-1]  # no-op, keep clarity
                      for lon, lat in st.session_state["section_line"]], dtype=float)
# (x,y) from helper returns tuple; build 2D array
line_xy_m = np.array([to_local_xy_m(lat, lon, lat0, lon0) for lon, lat in
                      [(p[1], p[0]) for p in st.session_state["section_line"]]], dtype=float)

pts_xy_m = np.array([to_local_xy_m(agg.loc[i,"Latitude"], agg.loc[i,"Longitude"], lat0, lon0)
                     for i in agg.index], dtype=float)
chain_ft, dist_ft, total_len_ft = project_chainage_to_polyline(pts_xy_m, line_xy_m)

mask = dist_ft <= corridor
sec = agg.loc[mask].copy()
sec["Chain_ft"] = chain_ft[mask]
sec = sec.sort_values("Chain_ft").reset_index(drop=True)

# Proposed projections
sec_prop = None
if prop is not None and not prop.empty:
    p_xy_m = np.array([to_local_xy_m(prop.loc[i,"Latitude"], prop.loc[i,"Longitude"], lat0, lon0)
                       for i in prop.index], dtype=float)
    p_chain, p_dist, _ = project_chainage_to_polyline(p_xy_m, line_xy_m)
    pmask = p_dist <= corridor
    sec_prop = prop.loc[pmask].copy()
    sec_prop["Chain_ft"] = p_chain[pmask]
    sec_prop = sec_prop.sort_values("Chain_ft").reset_index(drop=True)

# Build soil color mapping
palette = soil_palette()
soil_to_color = {}
sec["color"] = [color_for_soil(s, soil_to_color, palette) for s in sec["Soil"]]

# Plot disconnected columns
fig = go.Figure()
x = sec["Chain_ft"].to_numpy()
top = sec["Top_EL"].to_numpy()
bot = sec["Bottom_EL"].to_numpy()

# Columns (thick grey stems)
for xi, y1, y0 in zip(x, top, bot):
    fig.add_trace(go.Scatter(
        x=[xi, xi], y=[y0, y1], mode="lines",
        line=dict(color="rgba(0,0,0,0.8)", width=4),
        showlegend=False, hoverinfo="skip"
    ))

# Soil-colored ticks/markers at small vertical spacing (gives the “banded” look)
for soil_code, sub in sec.groupby("Soil"):
    xs = sub["Chain_ft"].to_numpy()
    ys = sub["Top_EL"].to_numpy()
    clr = color_for_soil(soil_code, soil_to_color, palette)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=9, color=clr, line=dict(color="black", width=0.5)),
        name=str(soil_code)
    ))

# Names + soil + N̄ as one-line labels
for _, r in sec.iterrows():
    label = str(r["Name"])
    soil_txt = r["Soil"] if isinstance(r["Soil"], str) else "Unknown"
    n_txt = f"N̄={r['N_text']}" if r["N_text"] else None
    extra = f" — {soil_txt}" + (f", {n_txt}" if n_txt else "")
    fig.add_annotation(
        x=r["Chain_ft"], y=r["Top_EL"], text=label + extra,
        showarrow=True, arrowhead=1, ax=0, ay=-28, font=dict(size=10)
    )

# Proposed points overlay (triangles)
if sec_prop is not None and not sec_prop.empty:
    fig.add_trace(go.Scatter(
        x=sec_prop["Chain_ft"], y=[sec["Top_EL"].median()] * len(sec_prop),
        mode="markers+text",
        marker=dict(size=10, symbol="triangle-up", color="#ea580c"),
        text=[str(n) for n in sec_prop["Name"]],
        textposition="top center",
        name="Proposed"
    ))

fig.update_layout(
    height=620,
    title=f"Disconnected Soil Columns — Length ≈ {total_len_ft:.0f} ft, corridor ±{corridor} ft",
    xaxis_title="Chainage (ft)",
    yaxis_title="Elevation (ft)",
    template="plotly_white",
    legend=dict(orientation="h"),
    hovermode="x"
)
st.plotly_chart(fig, use_container_width=True)
