# app.py — Map → Section slider → 2D + 3D (plan) profiles
# Simplified: always include all soil types (no multiselect)
# Optional toggles remain for soil code (ML/SM/etc.) and SPT N value.

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
EXISTING_TEXT_COLOR = "#1e88e5"
PROPOSED_TEXT_COLOR = "#e53935"
CIRCLE_RADIUS_PX, CIRCLE_STROKE_W = 10, 1
CIRCLE_STROKE, CIRCLE_FILL_OPACITY = "#ffffff", 0.95
LABEL_DX_PX, LABEL_DY_PX = 8, -10
TITLE_DEFAULT = "Soil Profile"
FIG_HEIGHT_IN = 22.0

SOIL_COLOR_MAP = {
    "Topsoil": "#ffffcb", "Water": "#00ffff",
    "CL": "#c5cae9", "CH": "#64b5f6", "CL-CH": "#fff176", "CL-ML": "#ef9a9a",
    "ML": "#ef5350", "MH": "#ffb74d",
    "GM": "#aed581", "GW": "#c8e6c9", "GC": "#00ff00", "GP": "#aaff32",
    "SM": "#76d7c4", "SP": "#ce93d8", "SC": "#81c784", "SW": "#ba68c8",
    "SM-SC": "#e1bee7", "SM-ML": "#dcedc8", "SC-CL": "#ffee58", "SC-SM": "#fff59d",
    "PWR": "#808080", "RF": "#929591", "Rock": "#c0c0c0",
}
ORDERED_SOIL_TYPES = ["Topsoil", "Water", "SM", "SC", "CL", "ML", "MH", "GM", "GP", "Rock", "PWR", "RF"]
RENAME_MAP = {
    'Bore Log': 'Borehole', 'Elevation From': 'Elevation_From', 'Elevation To': 'Elevation_To',
    'Soil Layer Description': 'Soil_Type', 'Latitude': 'Latitude', 'Longitude': 'Longitude', 'SPT N-Value': 'SPT',
}

def compute_spt_avg(value):
    if value is None or str(value).strip() in ("", "N/A"): return "N = N/A"
    nums = []
    for x in str(value).split(","):
        try: nums.append(float(x.strip().replace('"','')))
        except: pass
    return f"N = {round(sum(nums)/len(nums),2)}" if nums else "N = N/A"

def load_df_from_excel(f):
    df = pd.read_excel(f)
    df.rename(columns=RENAME_MAP, inplace=True)
    df['Soil_Type'] = df['Soil_Type'].astype(str).str.extract(r'\((.*?)\)').fillna(
        df['Soil_Type'].str.replace(r'^.*top\s*soil.*$', 'Topsoil', case=False, regex=True))
    df['Latitude'], df['Longitude'] = pd.to_numeric(df['Latitude'], errors='coerce'), pd.to_numeric(df['Longitude'], errors='coerce')
    df.dropna(subset=['Latitude','Longitude'], inplace=True)
    df['SPT_Label'] = df['SPT'].apply(compute_spt_avg)
    return df

def make_borehole_coords(df):
    return df.groupby('Borehole')[['Latitude','Longitude']].first().reset_index()

def total_geodesic_length_ft(line):
    return sum(geodesic((line.coords[i-1][1], line.coords[i-1][0]), (line.coords[i][1], line.coords[i][0])).feet for i in range(1,len(line.coords)))

def chainage_and_offset_ft(line, lat, lon):
    if len(line.coords)<2: return 0,0
    proj=line.project(Point(lon,lat)); frac=proj/line.length if line.length>0 else 0
    ch=total_geodesic_length_ft(line)*frac; near=line.interpolate(proj)
    off=geodesic((lat,lon),(near.y,near.x)).feet; return ch, off

def auto_y_limits(df,pad_ratio=0.05):
    if df.empty: return 0,1
    ymin,ymax=float(df['Elevation_To'].min()),float(df['Elevation_From'].max())
    pad=(ymax-ymin)*pad_ratio; return ymin-pad,ymax+pad

def dynamic_column_width(xp,default_width=60.0,min_width=8.0,fraction_of_min_gap=0.8):
    xs=sorted(xp.values());
    if len(xs)<2: return default_width
    gaps=[xs[i+1]-xs[i] for i in range(len(xs)-1) if xs[i+1]>xs[i]]
    if not gaps: return default_width
    return max(min_width,min(default_width,fraction_of_min_gap*min(gaps)))

def build_plotly_profile(df,ordered_bhs,x_positions,y_min,y_max,title,column_width,show_codes=False,show_spt=True,fig_height_px=1000):
    width=column_width if column_width else dynamic_column_width(x_positions); half=width/2
    def inner_font(w): return 12 if w>=60 else 11 if w>=45 else 10 if w>=30 else 9 if w>=18 else 8
    def bh_font(w): return 14 if w>=60 else 13 if w>=45 else 12 if w>=30 else 11 if w>=18 else 10
    shapes,annots,used=set(),[],set()
    for bh in ordered_bhs:
        bore=df[df['Borehole']==bh]
        if bore.empty: continue
        x=x_positions[bh]; top=bore['Elevation_From'].max()
        annots.append(dict(x=x,y=top+3,text=bh,showarrow=False,xanchor="center",font=dict(size=bh_font(width),family="Arial Black",color="#111")))
        for _,r in bore.iterrows():
            ef,et=float(r['Elevation_From']),float(r['Elevation_To']);soil=str(r['Soil_Type']);spt=str(r['SPT_Label']);
            color=SOIL_COLOR_MAP.get(soil,"#ccc");used.add(soil)
            shapes.add(bh)
            ann_text = soil if show_codes and not show_spt else spt if show_spt and not show_codes else f"{soil} ({spt})" if show_codes and show_spt else None
            annots.append(dict(x=x,y=(ef+et)/2,text=ann_text,showarrow=False,xanchor="center",yanchor="middle",font=dict(size=inner_font(width),family="Arial",color="#111"))) if ann_text else None
    fig=go.Figure();
    for s in [s for s in ORDERED_SOIL_TYPES if s in used]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(size=12,color=SOIL_COLOR_MAP.get(s,"#ccc")),name=s))
    xs=list(x_positions.values()); xmin,xmax=min(xs)-half,max(xs)+3*half
    fig.update_layout(title=title,height=fig_height_px,shapes=[dict(type="rect",x0=x-half,x1=x+half,y0=et,y1=ef,line=dict(color="#000",width=1.3),fillcolor=SOIL_COLOR_MAP.get(soil,"#ccc")) for bh in ordered_bhs for _,r in df[df['Borehole']==bh].iterrows() for soil,ef,et in [(r['Soil_Type'],r['Elevation_From'],r['Elevation_To'])]],annotations=annots,xaxis_title="Chainage (ft)",yaxis_title="Elevation (ft)",plot_bgcolor="white",legend=dict(x=1.02))
    fig.update_xaxes(range=[xmin,xmax]); fig.update_yaxes(range=[y_min,y_max]); return fig

# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Borehole Section Profile",layout="wide")
main_file=st.sidebar.file_uploader("MAIN borehole Excel (Elevations/Soils/SPT)",type=["xlsx","xls"])
if not main_file: st.info("Upload MAIN Excel to start."); st.stop()

df=load_df_from_excel(main_file); bh_coords=make_borehole_coords(df)

st.title("Section / Profile (ft) — Soil")
corridor_ft=st.slider("Corridor width (ft)",0,1000,200,10)

with st.expander("Label & Export",expanded=True):
    col1,col2=st.columns(2)
    with col1: show_codes=st.checkbox("Show soil code (ML/SM/...)",value=False)
    with col2: show_spt=st.checkbox("Show SPT N value",value=True)

if "section_line_coords" not in st.session_state: st.session_state["section_line_coords"]=None

st.info("Draw a polyline on the map first.")
