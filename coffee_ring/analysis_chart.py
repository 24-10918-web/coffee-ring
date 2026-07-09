"""
analysis_chart.py — 실험 + 시뮬레이션 통합 분석 차트

실험(링 너비 W)과 시뮬레이션(edge/center 비)을 한 화면에 겹쳐,
최적 에탄올 농도 구간을 시각적으로 도출한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

# ── 데이터 ──
phi = np.array([0, 30, 50, 70])
W_exp = np.array([0.45, 0.18, 0.11, 0.14])       # 실험 링 너비 (mm)
theta = np.array([19.5, 11.3, 8.1, 6.8])
sim_ratio = np.array([10.66, 2.09, 0.68, 0.05])  # 시뮬 edge/center

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Experiment + Simulation: Finding Optimal Ethanol Concentration",
             fontsize=14, fontweight='bold')

# ============================================================
# (좌) 실험 W + 시뮬 edge/center 겹치기
# ============================================================
ax = axes[0]
ax2 = ax.twinx()

# 실험 W (왼쪽 축)
l1 = ax.plot(phi, W_exp, 'o-', color="#BA7517", lw=2.2, ms=9,
             label="Experiment: ring width W", zorder=5)
ax.set_xlabel("ethanol concentration phi (%)", fontsize=11)
ax.set_ylabel("ring width W (mm)", color="#BA7517", fontsize=11)
ax.tick_params(axis='y', labelcolor="#BA7517")
ax.set_ylim(0, 0.5)

# E5 최소 표시
imin = np.argmin(W_exp)
ax.annotate("minimum W\n(E5, 50%)", xy=(phi[imin], W_exp[imin]),
            xytext=(phi[imin]-2, W_exp[imin]+0.12),
            fontsize=9, color="#BA7517", ha='center',
            arrowprops=dict(arrowstyle='->', color="#BA7517"))
# E7 반등 표시
ax.annotate("slight rebound\n(E7: wide faint deposit)",
            xy=(phi[3], W_exp[3]), xytext=(phi[3]-8, W_exp[3]+0.10),
            fontsize=9, color="#993C1D", ha='center',
            arrowprops=dict(arrowstyle='->', color="#993C1D"))

# 시뮬 edge/center (오른쪽 축, 로그)
l2 = ax2.plot(phi, sim_ratio, 's--', color="#378ADD", lw=2, ms=8,
              label="Simulation: edge/center ratio", zorder=4)
ax2.set_ylabel("simulation edge/center (log)", color="#378ADD", fontsize=11)
ax2.tick_params(axis='y', labelcolor="#378ADD")
ax2.set_yscale('log')
ax2.axhline(1.0, color="#378ADD", ls=':', lw=1, alpha=0.6)
ax2.text(2, 1.15, "ratio=1 (ring=center)", color="#378ADD", fontsize=8)

# 전환 구간 음영 (E5~E7: 링→중심 전환)
ax.axvspan(50, 70, alpha=0.08, color='red')
ax.text(60, 0.42, "ring -> center\ntransition", ha='center', fontsize=9,
        color='#993C1D')

# 최적 구간 표시 (E5 근처)
ax.axvspan(40, 55, alpha=0.15, color='green')
ax.text(47, 0.02, "optimal\nzone", ha='center', fontsize=9,
        color='#0F6E56', fontweight='bold')

lines = l1 + l2
ax.legend(lines, [x.get_label() for x in lines], loc='upper center', fontsize=9)
ax.set_title("(a) Ring width (exp) vs edge/center (sim)", fontsize=11)
ax.grid(alpha=0.3)

# ============================================================
# (우) 종합 해석: 침착 유형 다이어그램
# ============================================================
ax = axes[1]
ax.set_xlim(0, 70)
ax.set_ylim(0, 1)
ax.set_xlabel("ethanol concentration phi (%)", fontsize=11)
ax.set_yticks([])
ax.set_title("(b) Deposit type & optimal window", fontsize=11)

# 구간별 색/설명
zones = [
    (0, 25, "#D85A30", "Strong RING\n(edge deposit)"),
    (25, 45, "#BA7517", "Ring weakening"),
    (45, 58, "#1D9E75", "OPTIMAL\n(min W,\nlow center)"),
    (58, 70, "#534AB7", "CENTER deposit\n(wide, faint)"),
]
for x0, x1, col, label in zones:
    ax.axvspan(x0, x1, alpha=0.22, color=col)
    ax.text((x0+x1)/2, 0.5, label, ha='center', va='center',
            fontsize=9.5, color=col, fontweight='bold')

# 데이터 점 표시
for p, w, t in zip(phi, W_exp, theta):
    ax.plot([p], [0.12], 'ko', ms=6)
    ax.text(p, 0.05, f"E{p//10}\nW={w}", ha='center', fontsize=8)

ax.text(35, 0.9, "Optimal = lowest ring width BEFORE center deposit dominates",
        ha='center', fontsize=9.5, style='italic', color='#333',
        bbox=dict(boxstyle='round', facecolor='#FFF8E8', alpha=0.8))

plt.tight_layout()
plt.savefig('analysis_chart.png', dpi=130, bbox_inches='tight')
print("Saved: analysis_chart.png")
