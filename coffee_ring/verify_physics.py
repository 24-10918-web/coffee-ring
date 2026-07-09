"""
verify_physics.py — 물리 함수 시각적 검증

physics.py의 각 함수가 φ, r에 따라 물리적으로 타당하게
변하는지 그래프로 확인한다. (개발 2단계 전 검증용)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import physics as P

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Physics Module Verification (no tuning, literature values)',
             fontsize=14, fontweight='bold')

phi = np.linspace(0, 1, 200)
cases = {"E0": 0.0, "E3": 0.30, "E5": 0.50, "E7": 0.70}
colors = {"E0": "#378ADD", "E3": "#1D9E75", "E5": "#BA7517", "E7": "#D85A30"}

# --- (1) 표면장력 γ(φ) ---
ax = axes[0, 0]
ax.plot(phi, [P.surface_tension(p) * 1000 for p in phi], color="#534AB7", lw=2)
for label, p in cases.items():
    ax.scatter([p], [P.surface_tension(p) * 1000], color=colors[label],
               s=50, zorder=5, label=f"{label} ({int(p*100)}%)")
ax.set_xlabel("ethanol fraction phi")
ax.set_ylabel("surface tension [mN/m]")
ax.set_title("(1) Surface tension gamma(phi) — nonlinear")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- (2) dγ/dφ ---
ax = axes[0, 1]
ax.plot(phi, [P.dgamma_dphi(p) * 1000 for p in phi], color="#D4537E", lw=2)
ax.set_xlabel("ethanol fraction phi")
ax.set_ylabel("dgamma/dphi [mN/m]")
ax.set_title("(2) Surface tension gradient — steep at low phi")
ax.grid(alpha=0.3)
ax.axhline(0, color='gray', lw=0.5)

# --- (3) 점성 μ(φ) & 증발률 J(φ) ---
ax = axes[0, 2]
ax.plot(phi, [P.viscosity(p) * 1000 for p in phi], color="#0F6E56",
        lw=2, label="viscosity mu(phi)")
ax.set_xlabel("ethanol fraction phi")
ax.set_ylabel("viscosity [mPa·s]", color="#0F6E56")
ax.tick_params(axis='y', labelcolor="#0F6E56")
ax2 = ax.twinx()
ax2.plot(phi, [P.evaporation_rate(p) for p in phi], color="#993C1D",
         lw=2, ls='--', label="evap rate J0(phi)")
ax2.set_ylabel("relative evap rate", color="#993C1D")
ax2.tick_params(axis='y', labelcolor="#993C1D")
ax.set_title("(3) Viscosity (peak ~40%) & Evap rate (rises)")
ax.grid(alpha=0.3)

# --- (4) 액적 형상 h(r) 기판별 ---
ax = axes[1, 0]
R = 1e-3
r = np.linspace(0, R, 200)
for name, th, col in [("glass 19.5deg", 19.5, "#378ADD"),
                       ("OHP 50deg", 50.0, "#D85A30")]:
    h = P.droplet_height(r, R, th)
    ax.plot(r / R, h / R, color=col, lw=2, label=name)
ax.set_xlabel("r / R")
ax.set_ylabel("h / R")
ax.set_title("(4) Droplet shape h(r) by substrate")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- (5) 증발 플럭스 J(r) — 가장자리 발산 ---
ax = axes[1, 1]
r = np.linspace(1e-4, R * 0.999, 200)
for label, p in cases.items():
    J = P.evaporation_flux(r, R, 19.5, p)
    ax.plot(r / R, J, color=colors[label], lw=1.8, label=f"{label}")
ax.set_xlabel("r / R")
ax.set_ylabel("evap flux J(r)")
ax.set_title("(5) Evaporation flux — diverges at edge")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 8)

# --- (6) 알짜 속도 u_net(r) 케이스별 (핵심!) ---
ax = axes[1, 2]
r = np.linspace(1e-4, R * 0.999, 200)
for label, p in cases.items():
    u_net, u_cap, u_mar = P.net_velocity(r, R, 19.5, p)
    ax.plot(r / R, u_net, color=colors[label], lw=1.8, label=f"{label}")
ax.axhline(0, color='black', lw=0.8, ls=':')
ax.set_xlabel("r / R")
ax.set_ylabel("net velocity u_net(r)")
ax.set_title("(6) Net velocity: + outward(ring) / - inward(center)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.02, 0.95, "above 0 = ring\nbelow 0 = center deposit",
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='#FFF8E8', alpha=0.8))

plt.tight_layout()
plt.savefig('physics_verification.png', dpi=130, bbox_inches='tight')
print("Saved: physics_verification.png")
