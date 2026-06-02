# app.py — Borehole Mapping and Section Profiling Tool
# ✨ Fully updated
#   • Multi-sheet support for EXISTING and PROPOSED Excel files
#   • Distinct colors per sheet
#   • Sidebar toggles + color legend
#   • Integrated 2D and 3D profile plotting
#   • High-quality figure export options

from typing import Dict, List, Tuple, Optional
import io, math, re
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import LineString, Point, Polygon
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import Map, LayerControl
from folium.plugins import Draw
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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
# Higher DPI for the Matplotlib hatched profile display/export
MATPLOTLIB_DISPLAY_DPI = 220
MATPLOTLIB_EXPORT_DPI = 600

plt.rcParams["figure.dpi"] = MATPLOTLIB_DISPLAY_DPI
plt.rcParams["savefig.dpi"] = MATPLOTLIB_EXPORT_DPI

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
    "PWR": "#808080","RF": "#929591","Rock": "#c0c0c0","BASALT": "#bbf90f","BRECCIA": "#f5deb3","CHERT": "#a0522d",
    "CLAYSTONE": "#c7c10c","SANDY CLAYSTONE": "#ddd618","SILTY CLAYSTONE": "#cdc50a",
    "DIATOMITE": "#7bc8f6","DOLOMITE": "#000080", "LIMESTONE": "#006400","MUDSTONE": "#add8e6",
    "SANDSTONE": "#cb00f5","CLAYEY SANDSTONE": "#d94ff5","SILTY SANDSTONE": "#d648d7",
    "SHALE": "#13eac9",
    "CONGLOMERATE": "#0cdc53", "BASALT CONGLOMERATE": "#0cdc53",
    "SILTSTONE": "#ffad01","SANDY SILTSTONE": "#f2ab15",
}



# ── Hatch patterns per soil type (edit as you like) ──────────────────────────
SOIL_HATCH_MAP = {
    # Fine soils
    "CL": "....", "CH": "oooo", "ML": "////", "MH": "\\\\\\\\",
    "CL-ML": "xx", "CL-CH": "++", "ML-CL": "--",

    # Sands/Gravels
    "SM": "///", "SP": "xx", "SC": "\\\\", "SW": "++",
    "GM": "oo", "GP": "++", "GC": "xx", "GW": "..",
    "SM-SC": "/\\", "SM-ML": "..", "SC-CL": "xx", "SC-SM": "++",
    "SP-SM": "oo", "SP-SC": "xx", "SW-SM": "..",

    # Special
    "Topsoil": "....", "Water": "////",

    # Rock / lithology (examples)
    "Rock": "xx", "PWR": "xx", "RF": "++",
    "CLAYSTONE": "x.x.", "SANDY CLAYSTONE": "///", "SILTY CLAYSTONE": "++",
    "SILTSTONE": "***", "SANDY SILTSTONE": "*.*.",
    "CONGLOMERATE": "OO",
    "SANDSTONE": "...", "CLAYEY SANDSTONE": "/./.", "SILTY SANDSTONE": "|.|.|.", 
    "SHALE": "----",
}

def build_matplotlib_profile_hatched(
    df: pd.DataFrame,
    ordered_bhs: List[str],
    x_positions: Dict[str, float],
    y_min: float,
    y_max: float,
    title: str,
    column_width: Optional[float],
    lab_df: Optional[pd.DataFrame] = None,
    show_codes: bool = False,
    show_spt: bool = True,
    show_wc: bool = False,
    show_duw: bool = False,
    show_ucs: bool = False,
    show_ll: bool = False,
    show_pi: bool = False,
    figsize: Tuple[float, float] = (18, 10),
):
    """
    Matplotlib version of the soil profile with hatch patterns inside rectangles.
    """
    width = column_width if column_width is not None else dynamic_column_width(x_positions)
    half = width / 2.0

    xs = list(x_positions.values())
    # Add extra x-axis space so soil-code/lab labels beside the first and last
    # boreholes do not get clipped by the plot boundary.
    if xs:
        x_range = max(xs) - min(xs) if len(xs) > 1 else width
        x_label_pad = max(250.0, 4.0 * width, 0.06 * max(1.0, x_range))
        xmin = min(0.0, min(xs) - half - x_label_pad)
        xmax = max(xs) + half + x_label_pad
    else:
        xmin, xmax = -250.0, 250.0

    fig, ax = plt.subplots(figsize=figsize, dpi=MATPLOTLIB_DISPLAY_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Chainage along section (ft)", fontsize=16)
    ax.set_ylabel("Elevation (ft)", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, which="both", linewidth=0.6, alpha=0.65)

    used_types = set()

    # Water table markers (one per BH per groundwater reading type if available)
    water_points = {label: {"x": [], "y": [], "color": color, "mpl_marker": mpl_marker}
                    for _, label, color, _, mpl_marker in WATER_TABLE_PLOT_COLUMNS}

    for bh in ordered_bhs:
        bore = df[df["Borehole"] == bh]
        if bore.empty:
            continue

        x = x_positions[bh]
        top_el = float(bore["Elevation_From"].max())
        ax.text(x, top_el + 1.0, bh, ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Groundwater elevations (optional): during drilling and after drilling.
        for label, pt in collect_water_points_for_borehole(bore, x).items():
            water_points[label]["x"].extend(pt["x"])
            water_points[label]["y"].extend(pt["y"])

        # Soil layers
        for _, r in bore.iterrows():
            ef = float(r["Elevation_From"])
            et = float(r["Elevation_To"])
            soil = str(r["Soil_Type"]).strip()
            if soil:
                used_types.add(soil)

            face = SOIL_COLOR_MAP.get(soil, "#cccccc")
            hatch = SOIL_HATCH_MAP.get(soil, "////")  # default hatch if unknown

            rect = mpatches.Rectangle(
                (x - half, et),            # (x0, y0)
                width,                     # w
                (ef - et),                 # h
                facecolor=face,
                edgecolor="black",
                linewidth=1.3,
                hatch=hatch
            )
            ax.add_patch(rect)

            mid_y = (ef + et) / 2.0
            offset = max(4.0, half * 0.15)

            # Soil code outside LEFT
            if show_codes and soil:
                ax.text(x - half - offset, mid_y, soil, ha="right", va="center", fontsize=9)


        # SPT/lab labels from the separate Lab Test workbook at the exact sample elevations
        if lab_df is not None and not lab_df.empty:
            lab_bore = lab_df[lab_df["Borehole"].astype(str) == str(bh)].dropna(subset=["Sample_Elev"])
            lab_offset = max(4.0, half * 0.15)
            for _, lab in lab_bore.iterrows():
                label = format_lab_label(lab, show_spt=show_spt, show_wc=show_wc, show_duw=show_duw, show_ucs=show_ucs, show_ll=show_ll, show_pi=show_pi, sep="; ", style="mathtext")
                if not label:
                    continue
                y = float(lab["Sample_Elev"])
                ax.plot([x + half, x + half + lab_offset * 0.7], [y, y], color="black", linewidth=0.8)
                ax.text(x + half + lab_offset, y, label, ha="left", va="center", fontsize=8)

    # Plot groundwater markers after soil rectangles so they stay visible.
    water_handles = []
    for label, pts in water_points.items():
        if pts["x"]:
            ax.scatter(
                pts["x"], pts["y"],
                marker=pts["mpl_marker"], s=90,
                color=pts["color"], edgecolors="black", linewidths=0.8,
                zorder=5, label=label
            )
            water_handles.append(Line2D(
                [0], [0], marker=pts["mpl_marker"], color="none",
                markerfacecolor=pts["color"], markeredgecolor="black",
                markersize=9, label=label
            ))

    # Legend: show soil types present (ordered)
    ordered_present = [s for s in ORDERED_SOIL_TYPES if s in used_types]
    extra_present = sorted([s for s in used_types if s not in set(ORDERED_SOIL_TYPES)])
    legend_types = ordered_present + extra_present

    handles = water_handles.copy()
    for s in legend_types:
        handles.append(
            mpatches.Patch(
                facecolor=SOIL_COLOR_MAP.get(s, "#cccccc"),
                edgecolor="black",
                hatch=SOIL_HATCH_MAP.get(s, "////"),
                label=s
            )
        )
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    fig.tight_layout()
    return fig


def build_water_elevation_section_plot(
    df: pd.DataFrame,
    ordered_bhs: List[str],
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    figsize: Tuple[float, float] = (18, 6),
):
    """
    Standalone groundwater elevation plot by bore log name.
    The bore log order follows the drawn section/profile order.
    """
    if df is None or df.empty or not ordered_bhs:
        return None

    x_vals = list(range(len(ordered_bhs)))
    after_x, after_y = [], []
    during_x, during_y = [], []

    for i, bh in enumerate(ordered_bhs):
        bore = df[df["Borehole"].astype(str) == str(bh)]
        if bore.empty:
            continue

        if "Water_Elev_After" in bore.columns:
            vals = pd.to_numeric(bore["Water_Elev_After"], errors="coerce").dropna()
            if not vals.empty:
                after_x.append(i)
                after_y.append(float(vals.iloc[0]))

        if "Water_Elev_During" in bore.columns:
            vals = pd.to_numeric(bore["Water_Elev_During"], errors="coerce").dropna()
            if not vals.empty:
                during_x.append(i)
                during_y.append(float(vals.iloc[0]))

    all_y = after_y + during_y
    if not all_y:
        return None

    if y_min is None:
        y_min = math.floor((min(all_y) - 5.0) / 5.0) * 5.0
    if y_max is None:
        y_max = math.ceil((max(all_y) + 5.0) / 5.0) * 5.0
    if y_max <= y_min:
        y_min -= 5.0
        y_max += 5.0

    fig, ax = plt.subplots(figsize=figsize, dpi=MATPLOTLIB_DISPLAY_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if after_x:
        ax.scatter(
            after_x,
            after_y,
            s=70,
            marker="o",
            facecolor="orange",
            edgecolor="black",
            linewidth=1.0,
            label="Water Elevation After Drilling",
            zorder=4,
        )

    if during_x:
        ax.scatter(
            during_x,
            during_y,
            s=60,
            marker="D",
            facecolor="yellow",
            edgecolor="black",
            linewidth=1.0,
            label="Water Elevation During Drilling",
            zorder=5,
        )

    ax.set_xlim(-0.5, len(ordered_bhs) - 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Elevation (ft)", fontsize=14, fontweight="bold")
    ax.set_title("Borelog Name", fontsize=15, fontweight="bold", pad=16)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(ordered_bhs, rotation=90, fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, which="both", color="#d9d9d9", linewidth=0.8)

    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
    )

    fig.tight_layout()
    return fig

ORDERED_SOIL_TYPES = [
    "Topsoil", "Water",
    "SM", "SM-ML", "SM-SC", "SP-SM", "SP", "SW",
    "SC", "SC-CL", "SC-SM", "SP-SC","SW-SM",
    "CL", "CL-ML", "CL-CH", "CH",
    "ML", "MH", "ML-CL", 
    "GM", "GP-GM", "GP-GC", "GP", "GC", "GW",
    "Rock", "PWR", "RF", "BASALT CONGLOMERATE", "CONGLOMERATE", "BASALT", "BRECCIA", "CHERT",
    "CLAYSTONE", "SANDY CLAYSTONE", "SILTY CLAYSTONE", "DIATOMITE", "DOLOMITE", "LIMESTONE", "MUDSTONE",
    "SANDSTONE", "CLAYEY SANDSTONE", "SILTY SANDSTONE", "SHALE",
    "SILTSTONE", "SANDY SILTSTONE",
]
RENAME_MAP = {
    "Bore Log": "Borehole", "Elevation From": "Elevation_From",
    "Elevation To": "Elevation_To",
    "Soil Layer Description": "Soil_Type",
    "Latitude": "Latitude", "Longitude": "Longitude",
    "SPT N-Value": "SPT",
    # Keep the two groundwater readings separate so both can be shown in profiles.
    "Water Elevation During Drilling": "Water_Elev_During",
    "Water Elevation After Drilling": "Water_Elev_After",
}

# Profile symbols/colors for groundwater readings.
# The column names are created in standardize_water_table_columns().
WATER_TABLE_PLOT_COLUMNS = [
    ("Water_Elev_During", "Water During Drilling", "#00a6d6", "triangle-down", "v"),
    ("Water_Elev_After", "Water After Drilling", "#000000", "triangle-down", "v"),
]

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



def _to_number(value):
    """Convert values such as 12, 12.0, '12', or blanks to numeric/NaN."""
    return pd.to_numeric(value, errors="coerce")

def _fmt_num(value, decimals=1):
    """Clean numeric label formatting for profile annotations."""
    if pd.isna(value):
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _normalized_col_name(name: str) -> str:
    """Normalize a column name for tolerant matching across Excel header variants."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def standardize_water_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create separate numeric columns for groundwater elevations measured during and
    after drilling. This handles exact project headers plus common variations.
    """
    df = df.copy()
    normalized_to_original = {_normalized_col_name(c): c for c in df.columns}

    explicit_candidates = {
        "Water_Elev_During": [
            "waterelevationduringdrilling", "waterelevduringdrilling",
            "waterlevelduringdrilling", "groundwaterelevationduringdrilling",
            "groundwaterlevelduringdrilling", "duringdrillingwaterelevation",
        ],
        "Water_Elev_After": [
            "waterelevationafterdrilling", "waterelevafterdrilling",
            "waterlevelafterdrilling", "groundwaterelevationafterdrilling",
            "groundwaterlevelafterdrilling", "afterdrillingwaterelevation",
            "waterelev", "waterelevation",
        ],
    }

    for target_col, candidates in explicit_candidates.items():
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            continue
        for candidate in candidates:
            source_col = normalized_to_original.get(candidate)
            if source_col is not None:
                df[target_col] = pd.to_numeric(df[source_col], errors="coerce")
                break

    # Fallback for headers not covered above, e.g. "GW Elev. During Drilling".
    for c in list(df.columns):
        norm = _normalized_col_name(c)
        if not any(token in norm for token in ["water", "groundwater", "gw"]):
            continue
        if not any(token in norm for token in ["elev", "level"]):
            continue
        if "during" in norm and "Water_Elev_During" not in df.columns:
            df["Water_Elev_During"] = pd.to_numeric(df[c], errors="coerce")
        if "after" in norm and "Water_Elev_After" not in df.columns:
            df["Water_Elev_After"] = pd.to_numeric(df[c], errors="coerce")

    # Backward compatibility with older versions of this app.py that used one
    # generic Water_Elev column for after-drilling values.
    if "Water_Elev" in df.columns and "Water_Elev_After" not in df.columns:
        df["Water_Elev_After"] = pd.to_numeric(df["Water_Elev"], errors="coerce")

    return df


def collect_water_points_for_borehole(bore: pd.DataFrame, x: float) -> Dict[str, Dict[str, list]]:
    """Return plotted groundwater points for one borehole, split by reading type."""
    points: Dict[str, Dict[str, list]] = {}
    for col, label, color, plotly_symbol, mpl_marker in WATER_TABLE_PLOT_COLUMNS:
        if col not in bore.columns:
            continue
        wt_series = pd.to_numeric(bore[col], errors="coerce").dropna()
        if wt_series.empty:
            continue
        wt = float(wt_series.iloc[0])
        points[label] = {
            "x": [x],
            "y": [wt],
            "color": color,
            "plotly_symbol": plotly_symbol,
            "mpl_marker": mpl_marker,
            "source_col": col,
        }
    return points

def format_lab_label(row, show_spt=True, show_wc=False, show_duw=False, show_ucs=False, show_ll=False, show_pi=False, sep="; ", style="plain"):
    """Build one compact label shown at each SPT/lab sample depth."""
    parts = []

    if style == "html":
        gamma_d = "γ<sub>d</sub>"
        q_u = "q<sub>u</sub>"
    elif style == "mathtext":
        gamma_d = r"$\gamma_d$"
        q_u = r"$q_u$"
    else:
        gamma_d = "γd"
        q_u = "qu"

    if show_spt and "SPT" in row and pd.notna(row.get("SPT")):
        parts.append(f"N={_fmt_num(row.get('SPT'))}")
    if show_wc and "Water_Content" in row and pd.notna(row.get("Water_Content")):
        parts.append(f"w={_fmt_num(row.get('Water_Content'))}%")
    if show_duw and "Dry_Unit_Weight" in row and pd.notna(row.get("Dry_Unit_Weight")):
        parts.append(f"{gamma_d}={_fmt_num(row.get('Dry_Unit_Weight'))}pcf")
    if show_ucs and "UCS" in row and pd.notna(row.get("UCS")):
        parts.append(f"{q_u}={_fmt_num(row.get('UCS'), 2)}tsf")
    if show_ll and "LL" in row and pd.notna(row.get("LL")):
        parts.append(f"LL={_fmt_num(row.get('LL'))}")
    if show_pi and "PI" in row and pd.notna(row.get("PI")):
        parts.append(f"PI={_fmt_num(row.get('PI'))}")
    return sep.join(parts)

@st.cache_data(show_spinner=False)
def load_lab_multisheet(uploaded_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Load the separate Lab Test workbook.
    Expected columns include: Bore Log, Depth (ft), SPT N, Water Content (%),
    Dry Unit Weight (pcf), UCS (tsf), LL, and PI. Header spaces are ignored.
    """
    all_sheets = pd.read_excel(io.BytesIO(uploaded_bytes), sheet_name=None)
    result: Dict[str, pd.DataFrame] = {}
    rename = {
        "bore log": "Borehole",
        "borehole": "Borehole",
        "depth (ft)": "Depth_ft",
        "depth": "Depth_ft",
        "spt n": "SPT",
        "spt n-value": "SPT",
        "water content (%)": "Water_Content",
        "water content": "Water_Content",
        "dry unit weight (pcf)": "Dry_Unit_Weight",
        "dry unit weight": "Dry_Unit_Weight",
        "ucs (tsf)": "UCS",
        "ucs": "UCS",
        "ll": "LL",
        "liquid limit": "LL",
        "liquid limit (%)": "LL",
        "pi": "PI",
        "plasticity index": "PI",
        "plasticity index (%)": "PI",
    }

    for i, (sheet, df_lab) in enumerate(all_sheets.items()):
        if i >= 8:
            break
        if df_lab.empty:
            continue
        df_lab = df_lab.copy()
        df_lab.columns = [str(c).strip() for c in df_lab.columns]
        df_lab = df_lab.rename(columns={c: rename.get(str(c).strip().lower(), c) for c in df_lab.columns})
        if not {"Borehole", "Depth_ft"}.issubset(df_lab.columns):
            continue
        df_lab["Borehole"] = df_lab["Borehole"].astype(str).str.strip()
        for c in ["Depth_ft", "SPT", "Water_Content", "Dry_Unit_Weight", "UCS", "LL", "PI"]:
            if c in df_lab.columns:
                df_lab[c] = pd.to_numeric(df_lab[c], errors="coerce")
        keep = [c for c in ["Borehole", "Depth_ft", "SPT", "Water_Content", "Dry_Unit_Weight", "UCS", "LL", "PI"] if c in df_lab.columns]
        df_lab = df_lab.dropna(subset=["Borehole", "Depth_ft"])[keep].copy()
        if not df_lab.empty:
            df_lab["Lab_Sheet"] = sheet
            result[sheet] = df_lab
    return result

def attach_lab_elevations(lab_dict: Dict[str, pd.DataFrame], layer_df: pd.DataFrame) -> pd.DataFrame:
    """Compute sample elevation from borehole top elevation minus lab sample depth."""
    if not lab_dict or layer_df.empty:
        return pd.DataFrame()
    lab_df = pd.concat(lab_dict.values(), ignore_index=True)
    if lab_df.empty:
        return pd.DataFrame()
    top_map = layer_df.groupby("Borehole")["Elevation_From"].max().rename("Top_Elevation")
    lab_df = lab_df.merge(top_map, how="left", left_on="Borehole", right_index=True)
    lab_df["Sample_Elev"] = lab_df["Top_Elevation"] - lab_df["Depth_ft"]
    return lab_df

def normalize_soil_label(value):
    """
    Return a short soil label for plotting.
    - If the description contains a code in parentheses, use only that code.
    - If there is no parenthesized code, show only Topsoil/Top Soil when detected.
    - Otherwise return a blank label so long descriptions like ASPHALT/FILL are not plotted.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    match = re.search(r"\(([^()]*)\)", text)
    if match:
        code = match.group(1).strip()
        return code if code else ""

    if re.search(r"\btop\s*soil\b|\btopsoil\b", text, flags=re.IGNORECASE):
        return "Topsoil"

    return ""

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
        df = standardize_water_table_columns(df)

        # Ensure Latitude/Longitude columns exist and are numeric
        if not {"Latitude", "Longitude"}.issubset(df.columns):
            continue
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df = df.dropna(subset=["Latitude", "Longitude"]).copy()

        # Compute average SPT labels (N-values).
        # Some borehole layer files do not contain an SPT column because
        # SPT/lab data may be provided in a separate workbook. In that case,
        # keep this blank instead of calling .apply() on pd.NA.
        if "SPT" in df.columns:
            df["SPT_Label"] = df["SPT"].apply(compute_spt_avg)
        else:
            df["SPT_Label"] = ""

        # 🟡 Normalize soil names: extract only parenthesized soil codes.
        # If no code is present, only keep Topsoil; do not plot long descriptions
        # like "ASPHALT — black, hard surface" or "FILL — aggregate base...".
        soil_col = None
        for c in df.columns:
            if str(c).strip().lower() in ["soil_type", "soil layer description", "soil layer"]:
                soil_col = c
                break
        if soil_col:
            df[soil_col] = df[soil_col].apply(normalize_soil_label)
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
    y_values = [df['Elevation_To'], df['Elevation_From']]
    for water_col, _, _, _, _ in WATER_TABLE_PLOT_COLUMNS:
        if water_col in df.columns:
            vals = pd.to_numeric(df[water_col], errors="coerce").dropna()
            if not vals.empty:
                y_values.append(vals)
    y_all = pd.concat(y_values, ignore_index=True)
    y_min = float(y_all.min())
    y_max = float(y_all.max())
    rng = max(1.0, (y_max - y_min))
    pad = rng * pad_ratio
    return y_min - pad, y_max + pad


def _nice_step(rng: float, target: int = 10) -> float:
    """Choose a clean step size for grid/contour spacing over range rng."""
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



# ── Groundwater contour map helpers ─────────────────────────────────────────
def prepare_groundwater_contour_points(df: pd.DataFrame, water_col: str) -> pd.DataFrame:
    """Return one groundwater point per borehole for contour mapping."""
    required = {"Borehole", "Latitude", "Longitude", water_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    rows = []
    for bh, bore in df.groupby("Borehole"):
        vals = pd.to_numeric(bore[water_col], errors="coerce").dropna()
        if vals.empty:
            continue
        lat = pd.to_numeric(bore["Latitude"], errors="coerce").dropna()
        lon = pd.to_numeric(bore["Longitude"], errors="coerce").dropna()
        if lat.empty or lon.empty:
            continue
        rows.append({
            "Borehole": str(bh),
            "Latitude": float(lat.iloc[0]),
            "Longitude": float(lon.iloc[0]),
            "Water_Elev": float(vals.iloc[0]),
        })
    return pd.DataFrame(rows)


def idw_grid_interpolation(
    x, y, z,
    grid_size: int = 170,
    power: float = 2.0,
    pad_ratio: float = 0.08,
    clip_polygon_xy: Optional[Polygon] = None,
):
    """
    Inverse-distance weighted groundwater surface.
    If a drawn contour area is supplied, the interpolated surface is clipped to
    that polygon/rectangle. Otherwise it is clipped near the convex hull of the
    boreholes so the plot does not show a misleading rectangular contour field.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(z) < 3:
        return None, None, None

    xrng = max(float(x.max() - x.min()), 1.0)
    yrng = max(float(y.max() - y.min()), 1.0)

    if clip_polygon_xy is not None and not clip_polygon_xy.is_empty:
        minx, miny, maxx, maxy = clip_polygon_xy.bounds
        bxrng = max(float(maxx - minx), 1.0)
        byrng = max(float(maxy - miny), 1.0)
        xpad = bxrng * 0.02
        ypad = byrng * 0.02
        gx = np.linspace(float(minx - xpad), float(maxx + xpad), int(grid_size))
        gy = np.linspace(float(miny - ypad), float(maxy + ypad), int(grid_size))
    else:
        xpad = xrng * pad_ratio
        ypad = yrng * pad_ratio
        gx = np.linspace(float(x.min() - xpad), float(x.max() + xpad), int(grid_size))
        gy = np.linspace(float(y.min() - ypad), float(y.max() + ypad), int(grid_size))

    xx, yy = np.meshgrid(gx, gy)

    dx = xx[..., None] - x[None, None, :]
    dy = yy[..., None] - y[None, None, :]
    dist2 = dx * dx + dy * dy
    weights = 1.0 / np.maximum(dist2, 1.0e-9) ** (power / 2.0)
    zz = np.sum(weights * z[None, None, :], axis=2) / np.sum(weights, axis=2)

    try:
        from shapely.geometry import MultiPoint
        from shapely.prepared import prep
        if clip_polygon_xy is not None and not clip_polygon_xy.is_empty:
            clip_geom = clip_polygon_xy
        else:
            clip_geom = MultiPoint(list(zip(x, y))).convex_hull.buffer(max(xrng, yrng) * 0.04)
        prepared_clip = prep(clip_geom)
        mask = np.zeros_like(zz, dtype=bool)
        for r in range(zz.shape[0]):
            for c in range(zz.shape[1]):
                pt = Point(float(xx[r, c]), float(yy[r, c]))
                mask[r, c] = not (prepared_clip.contains(pt) or prepared_clip.touches(pt))
        zz = np.where(mask, np.nan, zz)
    except Exception:
        pass

    return gx, gy, zz


def build_groundwater_contour_figure(
    df: pd.DataFrame,
    water_col: str,
    title: str,
    section_line: Optional[LineString] = None,
    contour_area: Optional[Polygon] = None,
    grid_size: int = 170,
    contour_interval: Optional[float] = None,
) -> Optional[go.Figure]:
    """Create a plan-view groundwater contour map from borehole water elevations."""
    pts = prepare_groundwater_contour_points(df, water_col)
    if pts.empty or len(pts) < 3:
        return None

    # If the user drew a polygon/rectangle, only use boreholes inside that area.
    if contour_area is not None and not contour_area.is_empty:
        inside = []
        for _, r in pts.iterrows():
            pnt = Point(float(r["Longitude"]), float(r["Latitude"]))
            inside.append(contour_area.contains(pnt) or contour_area.touches(pnt))
        pts = pts.loc[inside].copy()
        if pts.empty or len(pts) < 3:
            return None

    lat0 = float(pts["Latitude"].mean())
    lon0 = float(pts["Longitude"].mean())
    xy = [latlon_to_local_xy_ft(float(r["Latitude"]), float(r["Longitude"]), lat0, lon0) for _, r in pts.iterrows()]
    pts["X_ft"] = [p[0] for p in xy]
    pts["Y_ft"] = [p[1] for p in xy]

    contour_area_xy = None
    if contour_area is not None and not contour_area.is_empty:
        try:
            area_xy = []
            for lon, lat in list(contour_area.exterior.coords):
                ax_ft, ay_ft = latlon_to_local_xy_ft(float(lat), float(lon), lat0, lon0)
                area_xy.append((ax_ft, ay_ft))
            contour_area_xy = Polygon(area_xy)
        except Exception:
            contour_area_xy = None

    gx, gy, gz = idw_grid_interpolation(
        pts["X_ft"], pts["Y_ft"], pts["Water_Elev"],
        grid_size=grid_size,
        clip_polygon_xy=contour_area_xy,
    )
    if gx is None or gz is None or np.all(np.isnan(gz)):
        return None

    zmin = float(np.nanmin(gz))
    zmax = float(np.nanmax(gz))
    if contour_interval is None or contour_interval <= 0:
        # Local fallback so the contour tool works even if the global _nice_step
        # helper is moved below this section or removed during edits/deployment.
        rng = max(zmax - zmin, 1.0)
        rough = rng / 10.0
        expv = math.floor(math.log10(rough)) if rough > 0 else 0
        frac = rough / (10 ** expv) if rough > 0 else 1.0
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
        contour_interval = nice * (10 ** expv)
    start = math.floor(zmin / contour_interval) * contour_interval
    end = math.ceil(zmax / contour_interval) * contour_interval

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=gx,
        y=gy,
        z=gz,
        colorscale="Viridis",
        contours=dict(
            start=start,
            end=end,
            size=contour_interval,
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=12, color="white"),
        ),
        line=dict(width=1.3, color="rgba(0,0,0,0.50)"),
        colorbar=dict(title="GW Elev. (ft)"),
        hovertemplate="Easting: %{x:.1f} ft<br>Northing: %{y:.1f} ft<br>GW Elev.: %{z:.2f} ft<extra></extra>",
        name="Groundwater elevation",
    ))

    fig.add_trace(go.Scatter(
        x=pts["X_ft"],
        y=pts["Y_ft"],
        mode="markers+text",
        marker=dict(size=12, color="white", line=dict(color="black", width=1.6)),
        text=pts["Borehole"],
        textposition="top center",
        textfont=dict(size=11, color="black"),
        customdata=np.stack([pts["Water_Elev"]], axis=-1),
        hovertemplate="%{text}<br>GW Elev.: %{customdata[0]:.2f} ft<extra></extra>",
        name="Boreholes",
    ))

    if contour_area is not None and not contour_area.is_empty:
        try:
            ax_line, ay_line = [], []
            for lon, lat in list(contour_area.exterior.coords):
                x, y = latlon_to_local_xy_ft(float(lat), float(lon), lat0, lon0)
                ax_line.append(x)
                ay_line.append(y)
            fig.add_trace(go.Scatter(
                x=ax_line,
                y=ay_line,
                mode="lines",
                line=dict(color="black", width=3, dash="dash"),
                name="Selected contour area",
                hoverinfo="skip",
            ))
        except Exception:
            pass

    if section_line is not None:
        try:
            sx, sy = [], []
            for lon, lat in section_line.coords:
                x, y = latlon_to_local_xy_ft(float(lat), float(lon), lat0, lon0)
                sx.append(x)
                sy.append(y)
            fig.add_trace(go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                line=dict(color="red", width=4),
                name="Section line",
                hoverinfo="skip",
            ))
        except Exception:
            pass

    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=20, color="black")),
        xaxis=dict(title="Easting (ft, local)", scaleanchor="y", scaleratio=1, showgrid=True, gridcolor="rgba(0,0,0,0.12)"),
        yaxis=dict(title="Northing (ft, local)", showgrid=True, gridcolor="rgba(0,0,0,0.12)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=780,
        margin=dict(l=70, r=120, t=80, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

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
lab_file = st.sidebar.file_uploader("Optional Lab Test Excel (SPT/lab values)", type=["xlsx", "xls"])

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

# Keep the drawn section line and groundwater contour area between Streamlit reruns.
if "section_line_coords" not in st.session_state:
    st.session_state["section_line_coords"] = None
if "contour_area_coords" not in st.session_state:
    st.session_state["contour_area_coords"] = None

# ── Load optional LAB TEST workbook (SPT + lab data) ────────────────────────
lab_df = pd.DataFrame()
if lab_file is not None:
    try:
        lab_dict = load_lab_multisheet(lab_file.getvalue())
        lab_df = attach_lab_elevations(lab_dict, df)
        matched = int(lab_df["Sample_Elev"].notna().sum()) if not lab_df.empty else 0
        if matched:
            st.sidebar.success(f"Loaded {matched} matching lab/SPT rows")
        elif not lab_df.empty:
            st.sidebar.warning("Lab file loaded, but no rows matched the selected borehole sheets.")
    except Exception as e:
        st.sidebar.warning(f"Could not read Lab Test workbook: {e}")
        lab_df = pd.DataFrame()

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

# Show the saved contour-area selection on the map after reruns.
if st.session_state.get("contour_area_coords"):
    try:
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in st.session_state["contour_area_coords"]],
            color="#ff4b4b",
            weight=3,
            fill=True,
            fill_opacity=0.16,
            tooltip="Groundwater contour area",
        ).add_to(fmap)
    except Exception:
        pass

# Drawing tools and layer toggle.
# Polyline = soil section/profile; Polygon/Rectangle = groundwater contour area.
Draw(
    draw_options={
        "polyline": {"shapeOptions": {"color": "#0000ff", "weight": 6}},
        "polygon": {"shapeOptions": {"color": "#ff4b4b", "weight": 3, "fillOpacity": 0.16}},
        "rectangle": {"shapeOptions": {"color": "#ff4b4b", "weight": 3, "fillOpacity": 0.16}},
        "circle": False,
        "marker": False,
        "circlemarker": False,
    },
    edit_options={"edit": True, "remove": True}
).add_to(fmap)
LayerControl(position="topright").add_to(fmap)

map_out = st_folium(fmap, height=600, use_container_width=True,
                    returned_objects=["last_active_drawing", "all_drawings"], key="map")

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

def extract_area_polygon(mo):
    def polygon_from_geom(geom):
        if geom.get("type") != "Polygon":
            return None
        coords = geom.get("coordinates", [])
        if not coords or len(coords[0]) < 4:
            return None
        poly = Polygon(coords[0])
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if poly.is_valid and not poly.is_empty else None

    lad = mo.get("last_active_drawing")
    if isinstance(lad, dict):
        poly = polygon_from_geom(lad.get("geometry", {}))
        if poly is not None:
            return poly
    if mo.get("all_drawings") and isinstance(mo["all_drawings"], dict):
        for feat in reversed(mo["all_drawings"].get("features", [])):
            poly = polygon_from_geom(feat.get("geometry", {}))
            if poly is not None:
                return poly
    return None

maybe_line = extract_linestring(map_out or {})
if maybe_line is not None:
    st.session_state["section_line_coords"] = list(map(list, maybe_line.coords))

maybe_area = extract_area_polygon(map_out or {})
if maybe_area is not None:
    st.session_state["contour_area_coords"] = list(map(list, maybe_area.exterior.coords))

# ── Groundwater contour maps in plan view ───────────────────────────────────
st.markdown("### Groundwater Contour Map — Plan View (optional)")
st.caption("Draw a red polygon or rectangle on the map to limit the groundwater contour area. Draw a blue polyline for the soil profile section.")
show_gw_contours = st.checkbox("Show groundwater contour map", value=False)

if show_gw_contours:
    gw_option_map = {
        "Water Elevation During Drilling": "Water_Elev_During",
        "Water Elevation After Drilling": "Water_Elev_After",
    }
    available_gw_options = [
        label for label, col in gw_option_map.items()
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]

    if not available_gw_options:
        st.info("No groundwater elevation data found for contour mapping.")
    else:
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            selected_gw_maps = st.multiselect(
                "Groundwater readings to contour",
                options=available_gw_options,
                default=available_gw_options,
            )
        with c2:
            gw_grid_size = st.slider("Contour smoothness", 80, 260, 170, 10)
        with c3:
            manual_interval = st.checkbox("Use manual contour interval", value=False)
            gw_contour_interval = None
            if manual_interval:
                gw_contour_interval = st.number_input("Interval (ft)", min_value=0.1, max_value=50.0, value=1.0, step=0.5)

        contour_scope = st.radio(
            "Contour area",
            options=["Drawn polygon/rectangle only", "Full selected dataset"],
            index=0 if st.session_state.get("contour_area_coords") else 1,
            horizontal=True,
        )

        contour_section_line = None
        if st.session_state.get("section_line_coords"):
            try:
                contour_section_line = LineString(st.session_state["section_line_coords"])
            except Exception:
                contour_section_line = None

        contour_area_polygon = None
        if contour_scope == "Drawn polygon/rectangle only":
            if st.session_state.get("contour_area_coords"):
                try:
                    contour_area_polygon = Polygon(st.session_state["contour_area_coords"])
                    if not contour_area_polygon.is_valid:
                        contour_area_polygon = contour_area_polygon.buffer(0)
                except Exception:
                    contour_area_polygon = None
            if contour_area_polygon is None or contour_area_polygon.is_empty:
                st.info("Draw a polygon or rectangle on the map to use the selected-area contour option.")

        missing_drawn_area = (
            contour_scope == "Drawn polygon/rectangle only"
            and (contour_area_polygon is None or contour_area_polygon.is_empty)
        )

        if not selected_gw_maps:
            st.info("Select at least one groundwater reading to show a contour map.")
        elif missing_drawn_area:
            pass
        else:
            for label in selected_gw_maps:
                col = gw_option_map[label]
                fig_contour = build_groundwater_contour_figure(
                    df=df,
                    water_col=col,
                    title=f"Groundwater Contour Map — {label}",
                    section_line=contour_section_line,
                    contour_area=contour_area_polygon,
                    grid_size=gw_grid_size,
                    contour_interval=gw_contour_interval,
                )
                if fig_contour is None:
                    st.warning(f"At least 3 boreholes with {label.lower()} are required to create a contour map.")
                else:
                    st.plotly_chart(
                        fig_contour,
                        use_container_width=True,
                        config={
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": f"groundwater_contour_{col.lower()}",
                                "scale": 4,
                            },
                        },
                    )
                    html = fig_contour.to_html(include_plotlyjs="cdn", full_html=True)
                    st.download_button(
                        f"Download interactive HTML — {label}",
                        data=html,
                        file_name=f"groundwater_contour_{col.lower()}.html",
                        mime="text/html",
                    )

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

colA, colB, colC, colD, colE, colF, colG = st.columns([1, 1, 1, 1, 1, 1, 1])
with colA:
    show_codes = st.checkbox("Show soil code (ML/SM/...)", value=False)
with colB:
    show_spt = st.checkbox("Show SPT N", value=True)
with colC:
    show_wc = st.checkbox("Show water content", value=False)
with colD:
    show_duw = st.checkbox("Show dry unit weight", value=False)
with colE:
    show_ucs = st.checkbox("Show UCS", value=False)
with colF:
    show_ll = st.checkbox("Show LL", value=False)
with colG:
    show_pi = st.checkbox("Show PI", value=False)

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

# ── 2D Profile Builder (Plotly) ─────────────────────────────────────────────
def build_plotly_profile(
    df: pd.DataFrame, ordered_bhs: List[str], x_positions: Dict[str, float],
    y_min: float, y_max: float, title: str,
    column_width: Optional[float],
    lab_df: Optional[pd.DataFrame] = None,
    show_codes: bool = False,
    show_spt: bool = True,
    show_wc: bool = False,
    show_duw: bool = False,
    show_ucs: bool = False,
    show_ll: bool = False,
    show_pi: bool = False,
    fig_height_px: int = 1000,
) -> go.Figure:
    # Width is either manual (passed) or auto from spacing
    width = column_width if column_width is not None else dynamic_column_width(x_positions)
    half = width / 2.0

    xs = list(x_positions.values())
    # Add extra x-axis space so soil-code/lab labels beside the first and last
    # boreholes do not get clipped by the plot boundary.
    if xs:
        x_range = max(xs) - min(xs) if len(xs) > 1 else width
        x_label_pad = max(250.0, 4.0 * width, 0.06 * max(1.0, x_range))
        xmin = min(0.0, min(xs) - half - x_label_pad)
        xmax = max(xs) + half + x_label_pad
    else:
        xmin, xmax = -250.0, 250.0

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

    # Collect water elevations per BH per groundwater reading type.
    water_points = {label: {"x": [], "y": [], "color": color, "plotly_symbol": plotly_symbol}
                    for _, label, color, plotly_symbol, _ in WATER_TABLE_PLOT_COLUMNS}
    lab_marker_x, lab_marker_y, lab_hover = [], [], []

    # Soil rectangles + labels
    for bh in ordered_bhs:
        bore = df[df["Borehole"] == bh]
        if bore.empty:
            continue
        x = x_positions[bh]
        top_el = bore["Elevation_From"].max()
        annotations.append(dict(
            x=x, y=top_el + 1.0, text=bh, showarrow=False,
            xanchor="center", yanchor="bottom",
            font=dict(size=bh_font, family="Arial Black", color="#111")
        ))

        for label, pt in collect_water_points_for_borehole(bore, x).items():
            water_points[label]["x"].extend(pt["x"])
            water_points[label]["y"].extend(pt["y"])

        for _, r in bore.iterrows():
            ef, et = float(r["Elevation_From"]), float(r["Elevation_To"])
            soil = str(r["Soil_Type"])
            color = SOIL_COLOR_MAP.get(soil, "#cccccc")
            if soil:
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
            if show_codes and soil:
                annotations.append(dict(
                    x=x - half - offset,
                    y=mid_y,
                    text=soil,
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    font=dict(size=label_font, family="Arial", color="#111"),
                ))


        # SPT/lab labels from the separate Lab Test workbook at the exact sample elevations
        if lab_df is not None and not lab_df.empty:
            lab_bore = lab_df[lab_df["Borehole"].astype(str) == str(bh)].dropna(subset=["Sample_Elev"])
            lab_offset = max(4.0, half * 0.15)
            for _, lab in lab_bore.iterrows():
                label = format_lab_label(lab, show_spt=show_spt, show_wc=show_wc, show_duw=show_duw, show_ucs=show_ucs, show_ll=show_ll, show_pi=show_pi, sep="; ", style="html")
                if not label:
                    continue
                y = float(lab["Sample_Elev"])
                lab_marker_x.append(x + half)
                lab_marker_y.append(y)
                lab_hover.append(f"{bh}<br>Depth: {_fmt_num(lab.get('Depth_ft'))} ft<br>{label}")
                annotations.append(dict(
                    x=x + half + lab_offset,
                    y=y,
                    text=label,
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    align="left",
                    font=dict(size=max(label_font - 2, 9), family="Arial", color="#111"),
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
    for label, pts in water_points.items():
        if pts["x"]:
            fig.add_trace(go.Scatter(
                x=pts["x"], y=pts["y"], mode="markers",
                marker=dict(
                    symbol=pts["plotly_symbol"], size=15,
                    color=pts["color"], line=dict(color="black", width=1)
                ),
                name=label,
                hovertemplate=f"{label}<br>Elevation: %{{y:.2f}} ft<extra></extra>"
            ))
    # Lab/SPT values are shown as text labels only.
    # Do not add sample point markers, so no open circles appear before labels.
    
    # ✅ ALWAYS apply layout (even if no water table exists)
    fig.update_layout(
        title=dict(text=title, font=dict(color="black", size=18)),
        font=dict(family="Inter, Arial, sans-serif"),
        xaxis=dict(title=dict(text="Chainage along section (ft)", font=dict(color="black", size=14))),
        yaxis=dict(title=dict(text="Elevation (ft)", font=dict(color="black", size=14))),
        shapes=grid_lines + soil_rects,
        annotations=annotations,
        height=fig_height_px,
        margin=dict(l=70, r=260, t=70, b=70),
        plot_bgcolor="white",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, bordercolor="#ddd", borderwidth=1),
    )
    
    fig.update_xaxes(range=[xmin, xmax], tickfont=dict(color="black", size=12), showgrid=False, zeroline=False)
    fig.update_yaxes(range=[y_min, y_max], tickfont=dict(color="black", size=12), showgrid=False, zeroline=False)
    return fig

# ── Generate 2D profile ─────────────────────────────────────────────────────
plot_df = df[df["Borehole"].isin(ordered_bhs)]
lab_plot_df = lab_df[lab_df["Borehole"].isin(ordered_bhs)].copy() if not lab_df.empty else pd.DataFrame()
if lab_file is not None:
    st.caption(f"Lab/SPT rows matching this section: **{len(lab_plot_df)}**")
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
    column_width=column_width_ft,
    lab_df=lab_plot_df,
    show_codes=show_codes,
    show_spt=show_spt,
    show_wc=show_wc,
    show_duw=show_duw,
    show_ucs=show_ucs,
    show_ll=show_ll,
    show_pi=show_pi,
    fig_height_px=fig_height_px
)
st.plotly_chart(
    fig2d, use_container_width=True,
    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "soil_profile", "scale": 4}}
)

st.markdown("### Soil Profile")

# Larger figure + higher DPI improves the on-screen profile quality.
# Width grows with the number of boreholes so labels stay readable.
profile_fig_width = max(18.0, min(34.0, 4.0 + 2.4 * max(1, len(ordered_bhs))))
profile_fig_height = 11.0

fig_hatched = build_matplotlib_profile_hatched(
    df=plot_df,
    ordered_bhs=ordered_bhs,
    x_positions=xpos,
    y_min=ymin_auto,
    y_max=ymax_auto,
    title=TITLE_DEFAULT,
    column_width=column_width_ft,   # uses same width logic as Plotly
    lab_df=lab_plot_df,
    show_codes=show_codes,
    show_spt=show_spt,
    show_wc=show_wc,
    show_duw=show_duw,
    show_ucs=show_ucs,
    show_ll=show_ll,
    show_pi=show_pi,
    figsize=(profile_fig_width, profile_fig_height)
)

# Save/export BEFORE displaying. Streamlit can clear the Matplotlib figure after display;
# saving after clear_figure=True can create blank PNG/SVG/PDF files.
# High-quality exports for reports. PNG is 600 dpi; SVG/PDF are vector.
png_buf = io.BytesIO()
fig_hatched.savefig(png_buf, format="png", dpi=MATPLOTLIB_EXPORT_DPI, bbox_inches="tight", facecolor="white")
png_buf.seek(0)
svg_buf = io.BytesIO()
fig_hatched.savefig(svg_buf, format="svg", bbox_inches="tight", facecolor="white")
svg_buf.seek(0)
pdf_buf = io.BytesIO()
fig_hatched.savefig(pdf_buf, format="pdf", bbox_inches="tight", facecolor="white")
pdf_buf.seek(0)

st.pyplot(fig_hatched, clear_figure=True)

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("Download high-quality PNG", png_buf, file_name="soil_profile_hatched_600dpi.png", mime="image/png")
with dl2:
    st.download_button("Download vector SVG", svg_buf, file_name="soil_profile_hatched.svg", mime="image/svg+xml")
with dl3:
    st.download_button("Download vector PDF", pdf_buf, file_name="soil_profile_hatched.pdf", mime="application/pdf")


st.markdown("### Groundwater Elevation Along Section")

water_fig_width = max(18.0, min(42.0, 0.35 * max(1, len(ordered_bhs))))
fig_water = build_water_elevation_section_plot(
    df=plot_df,
    ordered_bhs=ordered_bhs,
    y_min=ymin_auto,
    y_max=ymax_auto,
    figsize=(water_fig_width, 6.0),
)

if fig_water is None:
    st.info("No groundwater elevation data found for the selected section.")
else:
    water_png_buf = io.BytesIO()
    fig_water.savefig(
        water_png_buf,
        format="png",
        dpi=MATPLOTLIB_EXPORT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    water_png_buf.seek(0)

    st.pyplot(fig_water, clear_figure=True)
    st.download_button(
        "Download Groundwater Elevation Plot PNG",
        data=water_png_buf,
        file_name="groundwater_elevation_along_section.png",
        mime="image/png",
    )


# ── Lab/SPT property plots for selected boreholes in the drawn section ───────
def build_lab_property_matplotlib(
    lab_data: pd.DataFrame,
    boreholes: List[str],
    value_col: str,
    x_title: str,
    title: str,
    y_min: float,
    y_max: float,
    figsize: Tuple[float, float] = (5.6, 7.2),
):
    """Matplotlib scatter plot of one lab/SPT property against elevation."""
    fig, ax = plt.subplots(figsize=figsize, dpi=160)

    plotted = False
    if lab_data is not None and not lab_data.empty and value_col in lab_data.columns:
        for bh in boreholes:
            d = lab_data[lab_data["Borehole"].astype(str) == str(bh)].copy()
            if d.empty:
                continue
            d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
            d["Sample_Elev"] = pd.to_numeric(d["Sample_Elev"], errors="coerce")
            d = d.dropna(subset=[value_col, "Sample_Elev"])
            if d.empty:
                continue

            ax.scatter(
                d[value_col],
                d["Sample_Elev"],
                marker="o",
                s=42,
                facecolors="orange",
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
            )
            plotted = True

    ax.set_title(title, fontsize=12, fontweight="bold", pad=28)
    ax.set_xlabel(x_title, fontsize=11, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Elevation (ft)", fontsize=11)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.45)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if not plotted:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)

    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig, dpi: int = 600) -> bytes:
    """Return high-quality PNG bytes for a Matplotlib figure."""
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
    bio.seek(0)
    return bio.getvalue()


st.markdown("### Lab/SPT Plots for Boreholes in Section")
if lab_plot_df.empty:
    st.info("Upload the Lab Test Excel and draw/select a section with matching boreholes to show SPT/lab property plots.")
else:
    lab_plot_bhs_available = [bh for bh in ordered_bhs if str(bh) in set(lab_plot_df["Borehole"].astype(str))]
    selected_lab_plot_bhs = st.multiselect(
        "Select bore logs to plot",
        options=lab_plot_bhs_available,
        default=lab_plot_bhs_available,
    )

    if not selected_lab_plot_bhs:
        st.info("Select at least one bore log to show the SPT/lab plots.")
    else:
        def render_lab_property_plot(
            container,
            value_col: str,
            x_title: str,
            title: str,
            download_label: str,
            file_name: str,
            download_key: str,
            no_data_message: str,
        ):
            with container:
                if value_col in lab_plot_df.columns and lab_plot_df[value_col].notna().any():
                    fig_prop = build_lab_property_matplotlib(
                        lab_plot_df, selected_lab_plot_bhs, value_col,
                        x_title, title, ymin_auto, ymax_auto
                    )
                    prop_png = figure_to_png_bytes(fig_prop, dpi=600)
                    st.pyplot(fig_prop, clear_figure=True)
                    st.download_button(
                        download_label,
                        data=prop_png,
                        file_name=file_name,
                        mime="image/png",
                        key=download_key,
                    )
                else:
                    st.info(no_data_message)

        col_spt, col_duw, col_ucs = st.columns(3)
        render_lab_property_plot(
            col_spt, "SPT", "SPT N", "Elevation vs SPT N",
            "Download SPT Plot PNG", "elevation_vs_spt.png",
            "download_spt_plot_png", "No SPT N data found for the selected bore logs."
        )
        render_lab_property_plot(
            col_duw, "Dry_Unit_Weight", "γd (pcf)", "Elevation vs Dry Density",
            "Download Dry Density Plot PNG", "elevation_vs_dry_density.png",
            "download_dry_density_plot_png", "No dry density data found for the selected bore logs."
        )
        render_lab_property_plot(
            col_ucs, "UCS", "qu (tsf)", "Elevation vs UCS",
            "Download UCS Plot PNG", "elevation_vs_ucs.png",
            "download_ucs_plot_png", "No UCS data found for the selected bore logs."
        )

        col_wc, col_ll, col_pi = st.columns(3)
        render_lab_property_plot(
            col_wc, "Water_Content", "Water Content (%)", "Elevation vs Water Content",
            "Download Water Content Plot PNG", "elevation_vs_water_content.png",
            "download_water_content_plot_png", "No water content data found for the selected bore logs."
        )
        render_lab_property_plot(
            col_ll, "LL", "LL", "Elevation vs LL",
            "Download LL Plot PNG", "elevation_vs_ll.png",
            "download_ll_plot_png", "No LL data found for the selected bore logs."
        )
        render_lab_property_plot(
            col_pi, "PI", "PI", "Elevation vs PI",
            "Download PI Plot PNG", "elevation_vs_pi.png",
            "download_pi_plot_png", "No PI data found for the selected bore logs."
        )


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
            soil = str(r['Soil_Type']).strip()
            color = SOIL_COLOR_MAP.get(soil, '#cccccc')
            if soil:
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
