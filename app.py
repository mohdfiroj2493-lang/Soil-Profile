import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

st.set_page_config(page_title="Borehole Section & Soil Profile", layout="wide")

# --------------------------------------------------------------------------------------
# Helpers (no pyproj: use Web-Mercator-like planar coords for chainage calculations)
# --------------------------------------------------------------------------------------
R = 6378137.0  # meters, WGS84 semi-major

def lonlat_to_xy(lon_arr, lat_arr):
    lon = np.radians(np.asarray(lon_arr, float))
    lat = np.radians(np.asarray(lat_arr, float))
    x = R * lon
    # clip latitude to avoid singularities
    lat = np.clip(lat, -np.radians(85), np.radians(85))
    y = R * np.log(np.tan(np.pi/4 + lat/2))
    return np.column_stack([x, y])  # meters

FT_PER_M = 3.280839895

def pick(cols, *aliases):
    cols = [c.lower() for c in cols]
    for a in aliases:
        a = a.lower()
        if a in cols:
            return a
    return None

def project_chainage_to_polyline(points_xy_m, poly_xy_m):
    """Return chainage_ft, point_to_line_dist_ft, and total_length_ft along the drawn polyline."""
    if len(poly_xy_m) < 2:
        n = len(points_xy_m)
        return np.zeros(n), np.full(n, np.inf), 0.0

    seg_vecs = poly_xy_m[1:] - poly_xy_m[:-1]
    seg_len = np.linalg.norm(seg_vecs, axis=1)
    seg_len[seg_len == 0] = 1e-9
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    chain = np.zeros(len(points_xy_m))
    dist = np.full(len(points_xy_m), np.inf)

    for i, P in enumerate(points_xy_m):
        best_d = np.inf
        best_ch = 0.0
        for k, v in enumerate(seg_vecs):
            A = poly_xy_m[k]
            L2 = np.dot(v, v)
            t = np.clip(np.dot(P - A, v) / L2, 0.0, 1.0)
            Q = A + t * v
            d = np.linalg.norm(P - Q)
            if d < best_d:
                best_d = d
                best_ch = cum[k] + t * seg_len[k]
        chain[i] = best_ch
        dist[i] = best_d

    return chain * FT_PER_M, dist * FT_PER_M, cum[-1] * FT_PER_M

# simple USCS color map
USCS_COLORS = {
    "GW":"#8dd3c7","GP":"#ffffb3","GM":"#bebada","GC":"#fb8072",
    "SW":"#80b1d3","SP":"#fdb462","SM":"#b3de69","SC":"#fccde5",
    "ML":"#d9d9d9","CL":"#bc80bd","OL":"#ccebc5",
    "MH":"#ffed6f","CH":"#1f78b4","OH":"#33a02c",
    "PT":"#a6cee3","PEAT":"#a6cee3","FILL":"#aaaaaa"
}
def uscs_color(code):
    if not isinstance(code,str) or not code:
        return "#aaaaaa"
    code = code.upper().strip()
    # take first two letters if longer, e.g., "SM-SC" -> "SM"
    if '-' in code: code = code.split('-')[0]
    return USCS_COLORS.get(code, "#aaaaaa")

def readable(s):
    if s is None or (isinstance(s,float) and np.isnan(s)): return ""
    return str(s).replace("\n"," ").replace("\r"," ").strip()

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("📍 Section / Profile (ft)")
st.caption("Upload MAIN Excel, optionally a PROPOSED Excel. Draw a section (double-click to finish).")

c1, c2 = st.columns([2,2])
with c1:
    main_file = st.file_uploader("MAIN bore log Excel (required)", type=["xls","xlsx"], key="main_up")
with c2:
    prop_file = st.file_uploader("PROPOSED bore log Excel (optional)", type=["xls","xlsx"], key="prop_up")

# force map reset button (also clears previous drawn polylines)
if "map_reset" not in st.session_state:
    st.session_state.map_reset = 0
if st.button("🔄 Clear section / Reset map"):
    st.session_state.map_reset += 1

if not main_file:
    st.info("Upload your **MAIN** Excel to begin.")
    st.stop()

# --------------------------------------------------------------------------------------
# Read MAIN (and optional PROPOSED)
# --------------------------------------------------------------------------------------
raw = pd.read_excel(main_file)
raw.columns = [c.strip().lower() for c in raw.columns]

# Required cols
c_name  = pick(raw.columns, "name","boring id","hole id","id")
c_lat   = pick(raw.columns, "latitude","lat")
c_lon   = pick(raw.columns, "longitude","lon","long")
c_top   = pick(raw.columns, "top el","top elevation","ground elevation","boring elevation")
c_depth = pick(raw.columns, "depth","total depth","hole depth")

if not all([c_name,c_lat,c_lon,c_top,c_depth]):
    st.error("MAIN file must include: Name, Latitude, Longitude, Top EL, Depth (any reasonable header/alias).")
    st.stop()

data = pd.DataFrame({
    "Name": raw[c_name].astype(str),
    "Latitude": pd.to_numeric(raw[c_lat], errors="coerce"),
    "Longitude": pd.to_numeric(raw[c_lon], errors="coerce"),
    "Top_EL": pd.to_numeric(raw[c_top], errors="coerce"),
    "Depth": pd.to_numeric(raw[c_depth], errors="coerce")
}).dropna(subset=["Latitude","Longitude","Top_EL","Depth"])

data["Bottom_EL"] = data["Top_EL"] - data["Depth"]

# Optional PWR
c_pwr_el = pick(raw.columns, "pwr el","pwr elevation","weathered rock elevation")
c_pwr_d  = pick(raw.columns, "pwr depth","weathered rock depth")
data["PWR_EL"] = np.nan
if c_pwr_el: data["PWR_EL"] = pd.to_numeric(raw[c_pwr_el], errors="coerce")
if c_pwr_d:
    pwr_d = pd.to_numeric(raw[c_pwr_d], errors="coerce")
    data["PWR_EL"] = data["PWR_EL"].fillna(data["Top_EL"] - pwr_d)

# Soil code + SPT (for labels / legend). Accept many aliases.
c_soil = pick(raw.columns, "soil","soil type","uscs","uscs code","uscs group")
c_n    = pick(raw.columns, "n","spt n","spt n value","n value")
data["Soil"] = raw[c_soil].astype(str) if c_soil else ""
data["N"]    = pd.to_numeric(raw[c_n], errors="coerce") if c_n else np.nan

# Optional PROPOSED (only shown on map)
prop = None
if prop_file:
    pr = pd.read_excel(prop_file)
    pr.columns = [c.strip().lower() for c in pr.columns]
    c_nme = pick(pr.columns, "name","boring id","hole id","id")
    c_la  = pick(pr.columns, "latitude","lat")
    c_lo  = pick(pr.columns, "longitude","lon","long")
    if all([c_nme,c_la,c_lo]):
        prop = pd.DataFrame({
            "Name": pr[c_nme].astype(str),
            "Latitude": pd.to_numeric(pr[c_la], errors="coerce"),
            "Longitude": pd.to_numeric(pr[c_lo], errors="coerce"),
        }).dropna(subset=["Latitude","Longitude"])

# --------------------------------------------------------------------------------------
# Map (only the latest drawn line is used; double-click to finish; Clear button resets)
# --------------------------------------------------------------------------------------
st.subheader("🗺️ Draw section on the map")
center = (data["Latitude"].mean(), data["Longitude"].mean())
m = folium.Map(location=center, zoom_start=13, control_scale=True)

# base tiles
folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    name="Satellite",
    attr="Tiles © Esri"
).add_to(m)

# MAIN markers (single-line labels; no wrapping)
for _, r in data.iterrows():
    label = readable(r["Name"])
    folium.CircleMarker(
        [r["Latitude"], r["Longitude"]],
        radius=6, color="blue", fill=True, fill_opacity=0.8,
        tooltip=label
    ).add_to(m)
    folium.Marker(
        [r["Latitude"], r["Longitude"]],
        icon=folium.DivIcon(html=f"""
            <div style="font-size:11px;color:#111;white-space:nowrap;">
              {label}
            </div>
        """)
    ).add_to(m)

# PROPOSED markers (squares)
if prop is not None and not prop.empty:
    for _, r in prop.iterrows():
        folium.RegularPolygonMarker(
            [r["Latitude"], r["Longitude"]],
            number_of_sides=4, radius=6, color="darkred", fill=True, fill_opacity=0.9,
            tooltip=f"PROPOSED: {readable(r['Name'])}"
        ).add_to(m)

# draw control
from folium.plugins import Draw
Draw(
    export=False,
    draw_options={"polyline": True, "polygon": False, "rectangle": False,
                  "circle": False, "circlemarker": False, "marker": False},
    edit_options={"edit": True, "remove": True},
).add_to(m)
folium.LayerControl().add_to(m)

map_state = st_folium(
    m, height=520, use_container_width=True,
    key=f"map_{st.session_state.map_reset}",
    returned_objects=["last_active_drawing","all_drawings"]
)

# Pull the *latest* drawn line only
polyline = None
if map_state and map_state.get("last_active_drawing"):
    g = map_state["last_active_drawing"]
    if g and g.get("geometry",{}).get("type") == "LineString":
        polyline = g["geometry"]["coordinates"]  # [[lon, lat], ...]

# --------------------------------------------------------------------------------------
# Corridor & Profile style
# --------------------------------------------------------------------------------------
st.subheader("📈 Section / Profile (ft)")
c1, c2, c3 = st.columns([1,1,2])
with c1:
    corridor = st.slider("Corridor width (ft)", min_value=25, max_value=1000, value=200, step=25)
with c2:
    style = st.radio("Profile style", ["Soil types (disconnected columns)",
                                       "Soil & PWR bands (connected)"],
                     index=0)

# --------------------------------------------------------------------------------------
# Compute section members (within corridor) and plot
# --------------------------------------------------------------------------------------
if polyline is None:
    st.info("Draw a polyline on the map and double-click to finish. Use **Clear section** to remove previous lines.")
    st.stop()

# project to planar
XY_m = lonlat_to_xy(data["Longitude"].to_numpy(), data["Latitude"].to_numpy())
poly_xy_m = lonlat_to_xy(np.array(polyline)[:,0], np.array(polyline)[:,1])
chain_ft, dist_ft, total_len_ft = project_chainage_to_polyline(XY_m, poly_xy_m)

mask = dist_ft <= corridor
sec = data.loc[mask].copy()
sec["Chainage_ft"] = chain_ft[mask]
sec = sec.sort_values("Chainage_ft").reset_index(drop=True)

st.write(f"**Section along drawn line** (Length ≈ `{total_len_ft:.0f}` ft, corridor ±`{corridor}` ft)")

if sec.empty:
    st.warning("No borings fall within the corridor. Widen the corridor or redraw the line.")
    st.stop()

# --------------------------------- Plotly profile ------------------------------------
x   = sec["Chainage_ft"].to_numpy()
top = sec["Top_EL"].to_numpy()
bot = sec["Bottom_EL"].to_numpy()
pwr = sec["PWR_EL"].to_numpy()

fig = go.Figure()

# vertical posts
for xi, ytop, ybot in zip(x, top, bot):
    fig.add_trace(go.Scatter(x=[xi, xi], y=[ybot, ytop], mode="lines",
                             line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"))

# disconnected soil columns: color each post by its soil (if present) and annotate with soil + avg N
if style.startswith("Soil types"):
    # color per boring by dominant soil code
    soils = sec["Soil"].fillna("").astype(str).to_list()
    Ns    = sec["N"].to_numpy()

    # top & bottom outlines
    fig.add_trace(go.Scatter(
        x=x, y=top, mode="lines+markers",
        line=dict(color="black", width=1),
        marker=dict(size=5, color="black"),
        name="Top EL (ft)",
        hovertemplate="Top EL (ft): %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=bot, mode="lines",
        line=dict(color="black", width=1),
        name="Bottom EL (ft)",
        hovertemplate="Bottom EL (ft): %{y:.2f}<extra></extra>"
    ))

    # a colored dot at, say, 1/3 height to indicate soil class
    soil_y = bot + 0.33*(top-bot)
    soil_colors = [uscs_color(s) for s in soils]
    fig.add_trace(go.Scatter(
        x=x, y=soil_y, mode="markers",
        marker=dict(size=10, color=soil_colors, line=dict(color="black", width=0.5)),
        name="Soil",
        hovertemplate="Soil: %{text}<extra></extra>",
        text=soils
    ))

    # annotate name + soil + avg N (if available)
    for xi, yi, name, s, n in zip(x, top, sec["Name"], soils, Ns):
        txt = readable(name)
        if s:  txt += f" — {s}"
        if not np.isnan(n): txt += f" (N={int(round(n))})"
        fig.add_annotation(x=xi, y=yi, text=txt, showarrow=True, arrowhead=1, ax=0, ay=-25)

# connected Soil & PWR bands (like your screenshot)
else:
    def add_band(x_arr, y_upper, y_lower, color_rgba, name, showlegend=True, legendgroup=None):
        if len(x_arr) < 2: return
        xx = np.concatenate([x_arr, x_arr[::-1]])
        yy = np.concatenate([y_upper, y_lower[::-1]])
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines",
                                 line=dict(color="black", width=1),
                                 fill="toself", fillcolor=color_rgba, name=name,
                                 showlegend=showlegend, legendgroup=legendgroup, hoverinfo="skip"))

    lower_soil = np.where(np.isnan(pwr), bot, pwr)
    add_band(x, top, lower_soil, "rgba(34,197,94,0.55)", "Soil", True, "soil")

    m = ~np.isnan(pwr)
    if m.any():
        # split contiguous segments for PWR
        idx = np.where(m)[0]
        cuts = np.where(np.diff(idx) > 1)[0]
        segs = np.split(idx, cuts+1)
        first = True
        for s in segs:
            xs, yu, yl = x[s], pwr[s], bot[s]
            add_band(xs, yu, yl, "rgba(127,29,29,0.70)", "PWR", first, "pwr")
            first = False

    # outlines
    fig.add_trace(go.Scatter(x=x, y=top, mode="lines+markers",
                             line=dict(color="black", width=1),
                             marker=dict(size=5, color="black"),
                             name="Top EL (ft)",
                             hovertemplate="Top EL (ft): %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=bot, mode="lines",
                             line=dict(color="black", width=1),
                             name="Bottom EL (ft)",
                             hovertemplate="Bottom EL (ft): %{y:.2f}<extra></extra>"))

    # PWR dashed
    if m.any():
        first = True
        idx = np.where(m)[0]; cuts = np.where(np.diff(idx)>1)[0]; segs = np.split(idx, cuts+1)
        for s in segs:
            fig.add_trace(go.Scatter(x=x[s], y=pwr[s], mode="lines",
                                     line=dict(color="black", width=1, dash="dot"),
                                     name="PWR EL (ft)", showlegend=first, legendgroup="pwr", hoverinfo="skip"))
            first = False

fig.update_layout(
    template="plotly_white",
    xaxis_title="Chainage (ft)",
    yaxis_title="Elevation (ft)",
    hovermode="x unified",
    legend=dict(orientation="h"),
    height=620
)
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                 spikethickness=1, spikedash="dot")

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# 3D scatter of tops/bottoms (optional, lightweight)
# --------------------------------------------------------------------------------------
st.subheader("🌀 3D Borehole View (tops & bottoms, ft)")
limit3d = st.checkbox("Limit to section corridor", value=True)
ve = st.slider("Vertical exaggeration (display only)", 1.0, 6.0, 2.0, step=0.5)

data3d = sec if limit3d else data
XY3_m = lonlat_to_xy(data3d["Longitude"].to_numpy(), data3d["Latitude"].to_numpy())
X_ft = XY3_m[:,0] * FT_PER_M
Y_ft = XY3_m[:,1] * FT_PER_M
names = data3d["Name"].astype(str).to_numpy()

z_top = data3d["Top_EL"].to_numpy()
z_bot = data3d["Bottom_EL"].to_numpy()
z_pwr = data3d["PWR_EL"].to_numpy()

fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(x=X_ft, y=Y_ft, z=z_top, mode="markers",
    marker=dict(size=5), name="Top EL (ft)", text=names,
    hovertemplate="<b>%{text}</b><br>Top EL: %{z:.2f} ft<br>E: %{x:.0f}, N: %{y:.0f}<extra></extra>"
))
fig3d.add_trace(go.Scatter3d(x=X_ft, y=Y_ft, z=z_bot, mode="markers",
    marker=dict(size=4), name="Bottom EL (ft)", text=names,
    hovertemplate="<b>%{text}</b><br>Bottom EL: %{z:.2f} ft<br>E: %{x:.0f}, N: %{y:.0f}<extra></extra>"
))
m = ~np.isnan(z_pwr)
if m.any():
    fig3d.add_trace(go.Scatter3d(x=X_ft[m], y=Y_ft[m], z=z_pwr[m], mode="markers",
        marker=dict(size=4), name="PWR EL (ft)", text=names[m],
        hovertemplate="<b>%{text}</b><br>PWR EL: %{z:.2f} ft<br>E: %{x:.0f}, N: %{y:.0f}<extra></extra>"
    ))

fig3d.update_layout(
    height=600,
    scene=dict(
        xaxis_title="Easting (ft)", yaxis_title="Northing (ft)",
        zaxis_title=f"Elevation (ft) — {ve}×",
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=ve)
    ),
    legend=dict(orientation="h"),
    margin=dict(l=0, r=0, b=0, t=10),
)
st.plotly_chart(fig3d, use_container_width=True)
