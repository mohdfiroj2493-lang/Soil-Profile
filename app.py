import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------
# CONSTANTS
# ---------------------------------------------
GAMMA_W = 62.4  # pcf (unit weight of water)

# ---------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------
@dataclass
class SoilLayer:
    thickness: float      # ft
    phi_deg: float        # degrees
    cohesion: float       # psf
    gamma_dry: float      # pcf
    gamma_sat: float      # pcf


@dataclass
class Profile:
    layers: List[SoilLayer] = field(default_factory=list)
    water_table_depth: float = 0.0
    dz: float = 0.1  # ft

    def total_depth(self):
        return sum(L.thickness for L in self.layers)

    def layer_at_depth(self, z):
        s = 0.0
        for i, L in enumerate(self.layers):
            s_next = s + L.thickness
            if z <= s_next + 1e-6:
                return i
            s = s_next
        return len(self.layers) - 1

    def ka_kp(self, phi_deg):
        sinp = math.sin(math.radians(phi_deg))
        ka = (1 - sinp) / (1 + sinp)
        kp = (1 + sinp) / (1 - sinp)
        return ka, kp

    def effective_gamma(self, L: SoilLayer):
        return L.gamma_sat - GAMMA_W

    def compute(self):
        H = self.total_depth()
        z = np.arange(0, H + self.dz, self.dz)
        sigma_v_eff = np.zeros_like(z)
        u = np.zeros_like(z)

        running = 0.0
        for k in range(1, len(z)):
            dz_loc = z[k] - z[k-1]
            z_mid = 0.5 * (z[k] + z[k-1])
            i = self.layer_at_depth(z_mid)
            L = self.layers[i]
            if z_mid < self.water_table_depth:
                gamma_eff = L.gamma_dry
            else:
                gamma_eff = self.effective_gamma(L)
            running += gamma_eff * dz_loc
            sigma_v_eff[k] = running
            u[k] = GAMMA_W * max(0.0, z[k] - self.water_table_depth)

        sigma_h_a = np.zeros_like(z)
        sigma_h_p = np.zeros_like(z)

        for k in range(len(z)):
            L = self.layers[self.layer_at_depth(z[k])]
            ka, kp = self.ka_kp(L.phi_deg)
            c = L.cohesion
            s_a = ka * sigma_v_eff[k] - 2 * c * math.sqrt(ka)
            s_p = kp * sigma_v_eff[k] + 2 * c * math.sqrt(kp)
            sigma_h_a[k] = max(0, s_a + u[k])
            sigma_h_p[k] = max(0, s_p + u[k])

        return {"z": z, "active": sigma_h_a, "passive": sigma_h_p}


# ---------------------------------------------
# STREAMLIT APP
# ---------------------------------------------
st.set_page_config(page_title="Earth Pressure Diagram (English Units)", layout="wide")
st.title("🧱 Active & Passive Earth Pressure Diagram (English Units)")
st.markdown("### Rankine Theory with Cohesion, Dual Water Tables, and Excavation Depth")

st.sidebar.header("Input Parameters")

num_layers = st.sidebar.number_input("Number of soil layers", 1, 10, 3)
wt_active = st.sidebar.number_input("Water Table Depth (Active Side, ft)", 0.0, 100.0, 6.0, step=0.5)
wt_passive = st.sidebar.number_input("Water Table Depth (Passive Side, ft)", 0.0, 100.0, 8.0, step=0.5)
excavation_depth = st.sidebar.number_input("Excavation Depth (Passive Side, ft)", 0.0, 100.0, 10.0, step=0.5)

layers = []
st.sidebar.markdown("### Soil Layer Properties")
for i in range(int(num_layers)):
    st.sidebar.markdown(f"**Layer {i+1}**")
    t = st.sidebar.number_input(f"Thickness L{i+1} (ft)", 0.1, 100.0, 5.0)
    phi = st.sidebar.number_input(f"φ L{i+1} (°)", 0.0, 50.0, 30.0)
    c = st.sidebar.number_input(f"c L{i+1} (psf)", 0.0, 5000.0, 0.0)
    gd = st.sidebar.number_input(f"γ_dry L{i+1} (pcf)", 60.0, 150.0, 110.0)
    gs = st.sidebar.number_input(f"γ_sat L{i+1} (pcf)", 60.0, 150.0, 120.0)
    layers.append(SoilLayer(t, phi, c, gd, gs))

# ---------------------------------------------
# COMPUTE AND PLOT
# ---------------------------------------------
if st.sidebar.button("Compute"):
    profA = Profile(layers=layers, water_table_depth=wt_active)
    profP = Profile(layers=layers, water_table_depth=wt_passive)

    resA = profA.compute()
    resP = profP.compute()

    z = resA["z"]
    H = profA.total_depth()

    passive_adj = np.copy(resP["passive"])
    passive_adj[z < excavation_depth] = 0.0  # no passive above excavation
    if any(z == excavation_depth):
        passive_adj[z >= excavation_depth] -= passive_adj[z == excavation_depth][0]

    # ---------------------------------------------
    # DRAW FIGURE (Active right, Passive left)
    # ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # Wall at center
    ax.axvline(0, color="black", linewidth=5)

    # Passive (left)
    ax.plot(-passive_adj, z, color="red", linewidth=2, label="Passive Pressure (Left)")
    ax.fill_betweenx(z, 0, -passive_adj, color="salmon", alpha=0.4)

    # Active (right)
    ax.plot(resA["active"], z, color="blue", linewidth=2, label="Active Pressure (Right)")
    ax.fill_betweenx(z, 0, resA["active"], color="lightblue", alpha=0.4)

    # Excavation line
    ax.axhline(excavation_depth, color="k", linestyle="--", linewidth=1.2)
    ax.text(0, excavation_depth - 0.4, f"Excavation (z={excavation_depth:.1f} ft)",
            fontsize=9, ha="center", color="k")

    # Soil layers
    s = 0.0
    x_text = max(resA["active"]) * 1.1  # push labels further to right
    for i, L in enumerate(layers):
        s += L.thickness
        ax.axhline(s, color="k", linestyle="--", linewidth=0.8)
        mid_depth = s - L.thickness / 2
        ax.text(x_text, mid_depth,
                f"Layer {i+1}\nφ={L.phi_deg}°\nγ={L.gamma_dry} pcf",
                fontsize=9, color="black", va="center", ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Formatting
    ax.invert_yaxis()
    ax.set_xlabel("Lateral Pressure (psf)")
    ax.set_ylabel("Depth (ft)")
    ax.set_title("Active (Right) and Passive (Left) Earth Pressure Diagram (Positive Values Only)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    st.pyplot(fig)

    # ---------------------------------------------
    # DISPLAY SUMMARY
    # ---------------------------------------------
    st.markdown("### Model Summary")
    st.write(f"**Total Height:** {H:.2f} ft")
    st.write(f"**Water Table (Active):** {wt_active:.2f} ft")
    st.write(f"**Water Table (Passive):** {wt_passive:.2f} ft")
    st.write(f"**Excavation Depth:** {excavation_depth:.2f} ft")
    st.success("✅ Computation complete — soil layers labeled clearly on the right side.")
