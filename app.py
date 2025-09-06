import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from difflib import SequenceMatcher

st.set_page_config(page_title="Borehole Section & Soil Profile", layout="wide")

# ----------------------------- small helpers ---------------------------------
FT_PER_M = 3.280839895
R = 6378137.0  # spherical mercator

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower()) if isinstance(s, str) else ""

def best_guess(df_cols, aliases, default=None, min_ratio=0.55):
    cols = list(df_cols)
    # exact contains
    for a in aliases:
        for c in cols:
            if slug(a) in slug(c):
                return c
    # fuzzy
    best, score = default, 0.0
    for c in cols:
        for a in aliases:
            r = SequenceMatcher(None, slug(c), slug(a)).ratio()
            if r > score:
                score, best = r, c
    return best if score >= min_ratio else default

def lonlat_to_xy(lon_arr, lat_arr):
    lon = np.radians(np.asarray(lon_arr, float))
    lat = np.radians(np.asarray(lat_arr, float))
    lat = np.clip(lat, -np.radians(85), np.radians(85))
    x = R * lon
    y = R * np.log(np.tan(np.pi/4 + lat/2))
    return np.column_stack([x, y])  # meters

def project_chainage_to_polyline(points_xy_m, poly_xy_m):
    if poly_xy_m is None or len(poly_xy_m) < 2:
        n = len(points_xy_m)
        return np.zeros(n), np.full(n, np.inf), 0.0

    seg_vecs = poly_xy_m[1:] - poly_xy_m[:-1]
    seg_len = np.linalg.norm(seg_vecs, axis=1)
    seg_len[seg_len == 0] = 1e-9
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    chain = np.zeros(len(points_xy_m))
    dist = np.full(len(points_xy_m), np.inf)

    for i, P in enumerate(points_xy_m):
        best_d, best_ch = np.inf, 0.0
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
    if '-' in code: code = code.split('-')[0]
    return USCS_COLORS.get(code, "#aaaaaa")

def readable(s):
    if s is None: return ""
    s = str(s).strip().replace("\n"," ").replace("\r"," ")
    return s

# ----------------------------- UI: uploads -----------------------------------
st.title("📍 Section / Profile (ft)")
st.caption("Upload MAIN Excel, optionally PROPOSED. Map columns below, then draw a section (double-click to finish).")

c1, c2 = st.columns([1.2,1])
with c1:
    main_file = st.file_uploader("MAIN bore log Excel (required)", type=["xls","xlsx"], key="main_up")
with c2:
    prop_file = st.file_uploader("PROPOSED bore log Excel (optional)", type=["xls","xlsx"], key="prop_up")

if "map_reset" not in st.session_state:
    st.session_state.map_reset = 0
if st.button("🔄 Clear section / Reset map"):
    st.session_state.map_reset += 1

if not main_file:
    st.info("Upload MAIN Excel to continue.")
    st.stop()

# read file bytes once (so we can read multiple sheets without re-upload)
main_bytes = main_file.read()
main_xls = pd.ExcelFile(io.BytesIO(main_bytes))
sheet = st.selectbox("Select sheet (MAIN)", options=main_xls.sheet_names, index=0)
df = main_xls.parse(sheet_name=sheet)
st.write("Detected columns:", list(df.columns))

# ----------------------------- Column mapping (MAIN) --------------------------
col1, col2, col3 = st.columns(3)
aliases = {
    "name": ["name","boring id","borehole id","hole id","bh","bh id","id","bore id","log id"],
    "lat": ["latitude","lat","gps lat","wgs84 latitude","nad83 latitude"],
    "lon": ["longitude","lon","long","gps lon","wgs84 longitude","nad83 longitude"],
    "top": ["top el","top elevation","ground elevation","surface elevation","el top","top of ground","existing grade","gl elevation"],
    "depth": ["depth","total depth","hole depth","final depth","boring depth","drilled depth","log depth"],
    "bot": ["bottom el","bottom elevation","elevation bottom"],
    "pwr_el": ["pwr el","weathered rock elevation","wr el","refusal el","auger refusal el","rock el"],
    "pwr_d": ["pwr depth","wr depth","weathered rock depth","refusal depth"],
    "soil": ["soil","soil type","uscs","uscs code","uscs group"],
    "n": ["n","spt n","spt n value","n value","avg n"]
}
cols = list(df.columns)

with col1:
    c_name  = st.selectbox("MAIN: Name", options=["— Select —"]+cols,
                           index=(["— Select —"]+cols).index(best_guess(cols, aliases["name"], default="— Select —")))
    c_lat   = st.selectbox("MAIN: Latitude", options=["— Select —"]+cols,
                           index=(["— Select —"]+cols).index(best_guess(cols, aliases["lat"], default="— Select —")))
    c_lon   = st.selectbox("MAIN: Longitude", options=["— Select —"]+cols,
                           index=(["— Select —"]+cols).index(best_guess(cols, aliases["lon"], default="— Select —")))
with col2:
    c_top   = st.selectbox("MAIN: Top EL", options=["— Select —"]+cols,
                           index=(["— Select —"]+cols).index(best_guess(cols, aliases["top"], default="— Select —")))
    c_depth = st.selectbox("MAIN: Depth (ft)  ❖ If missing, map Bottom EL in the next box",
                           options=["— Select —"]+cols,
                           index=(["— Select —"]+cols).index(best_guess(cols, aliases["depth"], default="— Select —")))
    c_bot   = st.selectbox("MAIN: Bottom EL (optional; used if Depth missing)",
                           options=["(None)"]+cols,
                           index=(["(None)"]+cols).index(best_guess(cols, aliases["bot"], default="(None)")))
with col3:
    c_pwr_el = st.selectbox("MAIN: PWR EL (optional)", options=["(None)"]+cols,
                            index=(["(None)"]+cols).index(best_guess(cols, aliases["pwr_el"], default="(None)")))
    c_pwr_d  = st.selectbox("MAIN: PWR depth (optional)", options=["(None)"]+cols,
                            index=(["(None)"]+cols).index(best_guess(cols, aliases["pwr_d"], default="(None)")))
    c_soil   = st.selectbox("MAIN: Soil code (optional)", options=["(None)"]+cols,
                            index=(["(None)"]+cols).index(best_guess(cols, aliases["soil"], default="(None)")))
    c_n      = st.selectbox("MAIN: Avg SPT N (optional)", options=["(None)"]+cols,
                            index=(["(None)"]+cols).index(best_guess(cols, aliases["n"], default="(None)")))

def col_or_none(name):
    return None if name in ("— Select —","(None)", None) else name

c_name  = col_or_none(c_name)
c_lat   = col_or_none(c_lat)
c_lon   = col_or_none(c_lon)
c_top   = col_or_none(c_top)
c_depth = col_or_none(c_depth)
c_bot   = col_or_none(c_bot)
c_pwr_el= col_or_none(c_pwr_el)
c_pwr_d = col_or_none(c_pwr_d)
c_soil  = col_or_none(c_soil)
c_n     = col_or_none(c_n)

# Validate required fields
missing = [lab for lab,val in [("Name",c_name),("Latitude",c_lat),("Longitude",c_lon),("Top EL",c_top)] if val is None]
if c_depth is None and c_bot is None:
    missing.append("Depth or Bottom EL")
if missing:
    st.error("Please map the following required fields: " + ", ".join(missing))
    st.stop()

# Build normalized dataframe
data = pd.DataFrame({
    "Name": df[c_name].astype(str),
    "Latitude": pd.to_numeric(df[c_lat], errors="coerce"),
    "Longitude": pd.to_numeric(df[c_lon], errors="coerce"),
    "Top_EL": pd.to_numeric(df[c_top], errors="coerce")
})
if c_depth:
    data["Depth"] = pd.to_numeric(df[c_depth], errors="coerce")
elif c_bot:
    bot = pd.to_numeric(df[c_bot], errors="coerce")
    data["Depth"] = data["Top_EL"] - bot

data = data.dropna(subset=["Latitude","Longitude","Top_EL","Depth"])
data["Bottom_EL"] = data["Top_EL"] - data["Depth"]

# Optional extras
data["PWR_EL"] = np.nan
if c_pwr_el:
    data["PWR_EL"] = pd.to_numeric(df[c_pwr_el], errors="coerce")
if c_pwr_d:
    pwr_d = pd.to_numeric(df[c_pwr_d], errors="coerce")
    data["PWR_EL"] = data["PWR_EL"].fillna(data["Top_EL"] - pwr_d)

data["Soil"] = df[c_soil].astype(str) if c_soil else ""
data["N"]    = pd.to_numeric(df[c_n], errors="coerce") if c_n else np.nan

# ----------------------------- Optional PROPOSED ------------------------------
prop = None
if prop_file is not None:
    pb = prop_file.read()
    px = pd.ExcelFile(io.BytesIO(pb))
    psheet = st.selectbox("Select sheet (PROPOSED)", options=px.sheet_names, index=0)
    pr = px.parse(sheet_name=psheet)
    pr_cols = list(pr.columns)
    pr_name = best_guess(pr_cols, aliases["name"])
    pr_lat  = best_guess(pr_cols, aliases["lat"])
    pr_lon  = best_guess(pr_cols, aliases["lon"])
    with st.expander("Map PROPOSED columns (optional)"):
        pr_name = st.selectbox("PROPOSED: Name", ["— Select —"]+pr_cols, index=(["— Select —"]+pr_cols).index(pr_name or "— Select —"))
        pr_lat  = st.selectbox("PROPOSED: Latitude", ["— Select —"]+pr_cols, index=(["— Select —"]+pr_cols).index(pr_lat or "— Select —"))
        pr_lon  = st.selectbox("PROPOSED: Longitude", ["— Select —"]+pr_cols, index=(["— Select —"]+pr_cols).index(pr_lon or "— Select —"))
    if pr_name != "— Select —" and pr_lat != "— Select —" and pr_lon != "— Select —":
        prop = pd.DataFrame({
            "Name": pr[pr_name].astype(str),
            "Latitude": pd.to_numeric(pr[pr_lat], errors="coerce"),
            "Longitude": pd.to_numeric(pr[pr_lon], errors="coerce"),
        }).dropna(subset=["Latitude","Longitude"])

# ----------------------------- Map & drawing ---------------------------------
st.subheader("🗺️ Draw section on the map")
center = (data["Latitude"].mean(), data["Longitude"].mean())
m = folium.Map(location=center, zoom_start=13, control_scale=True)
folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    name="Satellite", attr="Tiles © Esri"
).add_to(m)

# main pins + single-line labels
for _, r in data.iterrows():
    lab = readable(r["Name"])
    folium.CircleMarker([r["Latitude"], r["Longitude"]],
                        radius=6, color="blue", fill=True, fill_opacity=0.9,
                        tooltip=lab).add_to(m)
    folium.Marker([r["Latitude"], r["Longitude"]],
                  icon=folium.DivIcon(html=f'<div style="font-size:11px;color:#111;white-space:nowrap;">{lab}</div>')).add_to(m)

# proposed pins (squares)
if prop is not None and not prop.empty:
    for _, r in prop.iterrows():
        folium.RegularPolygonMarker([r["Latitude"], r["Longitude"]],
                                    number_of_sides=4, radius=6, color="darkred",
                                    fill=True, fill_opacity=0.9,
                                    tooltip=f"PROPOSED: {readable(r['Name'])}").add_to(m)

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

polyline = None
if map_state and map_state.get("last_active_drawing"):
    g = map_state["last_active_drawing"]
    if g and g.get("geometry",{}).get("type") == "LineString":
        polyline = g["geometry"]["coordinates"]  # [[lon,lat],...]

# ----------------------------- Section settings -------------------------------
st.subheader("📈 Section / Profile (ft)")
cA, cB, cC = st.columns([1,1,2])
with cA:
    corridor = st.slider("Corridor width (ft)", 25, 1000, 200, step=25)
with cB:
    style = st.radio("Profile style", ["Soil types (disconnected columns)",
                                       "Soil & PWR bands (connected)"], index=0)

if polyline is None:
    st.info("Draw a polyline on the map and **double-click** to finish. Use **Clear section** to remove previous lines.")
    st.stop()

# ----------------------------- Build section ----------------------------------
XY_m = lonlat_to_xy(data["Longitude"].to_numpy(), data["Latitude"].to_numpy())
poly_xy_m = lonlat_to_xy(np.array(polyline)[:,0], np.array(polyline)[:,1])
chain_ft, dist_ft, total_len_ft = project_chainage_to_polyline(XY_m, poly_xy_m)

mask = dist_ft <= corridor
sec = data.loc[mask].copy()
sec["Chainage_ft"] = chain_ft[mask]
sec = sec.sort_values("Chainage_ft").reset_index(drop=True)

st.write(f"**Section along drawn line** (Length ≈ `{total_len_ft:.0f}` ft, corridor ±`{corridor}` ft)")
if sec.empty:
    st.warning("No borings within the corridor. Widen corridor or redraw the line.")
    st.stop()

x   = sec["Chainage_ft"].to_numpy()
top = sec["Top_EL"].to_numpy()
bot = sec["Bottom_EL"].to_numpy()
pwr = sec["PWR_EL"].to_numpy()

fig = go.Figure()

# posts
for xi, ytop, ybot in zip(x, top, bot):
    fig.add_trace(go.Scatter(x=[xi, xi], y=[ybot, ytop], mode="lines",
                             line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"))

if style.startswith("Soil types"):
    soils = sec["Soil"].fillna("").astype(str).to_list()
    Ns    = sec["N"].to_numpy()
    fig.add_trace(go.Scatter(x=x, y=top, mode="lines+markers",
                             line=dict(color="black", width=1),
                             marker=dict(size=5, color="black"),
                             name="Top EL (ft)"))
    fig.add_trace(go.Scatter(x=x, y=bot, mode="lines",
                             line=dict(color="black", width=1),
                             name="Bottom EL (ft)"))
    # colored dot on each post
    soil_y = bot + 0.33*(top-bot)
    fig.add_trace(go.Scatter(
        x=x, y=soil_y, mode="markers",
        marker=dict(size=10, color=[uscs_color(s) for s in soils],
                    line=dict(color="black", width=0.5)),
        name="Soil", text=soils, hovertemplate="Soil: %{text}<extra></extra>"
    ))
    # label: Name — Soil (N=)
    for xi, yi, nm, s, n in zip(x, top, sec["Name"], soils, Ns):
        txt = readable(nm)
        if s: txt += f" — {s}"
        if not (isinstance(n,float) and np.isnan(n)):
            try: txt += f" (N={int(round(float(n)))})"
            except: pass
        fig.add_annotation(x=xi, y=yi, text=txt, showarrow=True, arrowhead=1, ax=0, ay=-25)

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
        idx = np.where(m)[0]
        cuts = np.where(np.diff(idx) > 1)[0]
        segs = np.split(idx, cuts+1)
        first = True
        for s in segs:
            add_band(x[s], pwr[s], bot[s], "rgba(127,29,29,0.70)", "PWR", first, "pwr")
            first = False

    fig.add_trace(go.Scatter(x=x, y=top, mode="lines+markers",
                             line=dict(color="black", width=1),
                             marker=dict(size=5, color="black"),
                             name="Top EL (ft)"))
    fig.add_trace(go.Scatter(x=x, y=bot, mode="lines",
                             line=dict(color="black", width=1),
                             name="Bottom EL (ft)"))
    if (~np.isnan(pwr)).any():
        first = True
        idx = np.where(~np.isnan(pwr))[0]; cuts = np.where(np.diff(idx)>1)[0]; segs = np.split(idx, cuts+1)
        for s in segs:
            fig.add_trace(go.Scatter(x=x[s], y=pwr[s], mode="lines",
                                     line=dict(color="black", width=1, dash="dot"),
                                     name="PWR EL (ft)", showlegend=first, legendgroup="pwr"))
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

# ----------------------------- 3D (optional) ---------------------------------
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
