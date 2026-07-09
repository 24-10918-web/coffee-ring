"""
fine_chart.py — 촘촘한 스윕 결과 시각화

fine_sweep.py가 저장한 sweep_data.npz를 읽어 정밀 분석 차트를 그린다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

d = np.load("sweep_data.npz")
phi = d["phi"] * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Fine-grained Simulation Sweep (2.5% steps) — Precise Transition",
             fontsize=14, fontweight='bold')

# ============================================================
# (좌) edge/center 곡선 + 전환점
# ============================================================
ax = axes[0]
ax.plot(phi, d["glass_ratio"], 'o-', color="#378ADD", ms=4, lw=1.8,
        label="glass (19.5deg)")
ax.fill_between(phi,
                np.array(d["glass_ratio"]) - np.array(d["glass_std"]),
                np.array(d["glass_ratio"]) + np.array(d["glass_std"]),
                color="#378ADD", alpha=0.15)
ax.plot(phi, d["ohp_ratio"], 's--', color="#D85A30", ms=4, lw=1.8,
        label="OHP (50deg)")

ax.axhline(1.0, color='gray', ls=':', lw=1.2)
ax.text(1, 1.15, "edge/center = 1 (ring = center)", fontsize=8, color='gray')

# 전환점 표시
tg, to = float(d["tg"]), float(d["to"])
ax.axvline(tg, color="#378ADD", ls=':', lw=1, alpha=0.7)
ax.annotate(f"glass\n{tg:.0f}%", xy=(tg, 1.0), xytext=(tg-9, 0.35),
            fontsize=9, color="#378ADD", ha='center',
            arrowprops=dict(arrowstyle='->', color="#378ADD"))
ax.axvline(to, color="#D85A30", ls=':', lw=1, alpha=0.7)
ax.annotate(f"OHP\n{to:.0f}%", xy=(to, 1.0), xytext=(to+2, 3),
            fontsize=9, color="#D85A30", ha='center',
            arrowprops=dict(arrowstyle='->', color="#D85A30"))
# 차이 강조
ax.annotate("", xy=(tg, 0.5), xytext=(to, 0.5),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=1))
ax.text((tg+to)/2, 0.55, f"{to-tg:.0f}%p\ndifference", ha='center',
        fontsize=8, color='#444')

# 전이대(MIXED) 음영
ax.axvspan(40, 47.5, alpha=0.12, color='#BA7517')
ax.text(43.7, 6, "MIXED\nzone", ha='center', fontsize=8, color='#8a5a00')

ax.set_yscale('log')
ax.set_xlabel("ethanol concentration phi (%)", fontsize=11)
ax.set_ylabel("edge/center ratio (log)", fontsize=11)
ax.set_title("(a) edge/center vs phi — transition at ~44%", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')

# ============================================================
# (우) edge% / center% 교차
# ============================================================
ax = axes[1]
ax.plot(phi, np.array(d["glass_edge"]) * 100, '-', color="#378ADD", lw=2,
        label="edge% (glass)")
ax.plot(phi, np.array(d["glass_center"]) * 100, '-', color="#D85A30", lw=2,
        label="center% (glass)")

# 교차점 찾기
ge = np.array(d["glass_edge"]) * 100
gc = np.array(d["glass_center"]) * 100
# edge가 center의 2배가 되는 지점 등 참고선
ax.fill_between(phi, ge, gc, where=(ge > gc), alpha=0.08, color="#378ADD")

# 최적 구간 (전환점 근방, W 최소 대응)
ax.axvspan(42, 50, alpha=0.15, color='#1D9E75')
ax.text(46, 72, "glass\noptimal", ha='center', fontsize=8.5,
        color='#0F6E56', fontweight='bold')
# OHP 최적 구간 (더 높은 농도)
ax.axvspan(60, 68, alpha=0.12, color='#D85A30')
ax.text(64, 72, "OHP\noptimal", ha='center', fontsize=8.5,
        color='#993C1D', fontweight='bold')

# 실험 데이터 점 (참고: E5=50%가 W 최소)
ax.axvline(50, color='#BA7517', ls='--', lw=1.2, alpha=0.7)
ax.text(50.5, 88, "E5 (exp W min)", fontsize=8, color='#8a5a00', rotation=90, va='top')

ax.set_xlabel("ethanol concentration phi (%)", fontsize=11)
ax.set_ylabel("particle fraction (%)", fontsize=11)
ax.set_title("(b) edge vs center fraction (glass)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("fine_chart.png", dpi=130, bbox_inches='tight')
print("Saved: fine_chart.png")
