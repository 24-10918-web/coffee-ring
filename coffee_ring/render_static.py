"""
render_static.py — 4분할 레이아웃 정적 렌더 (검증용)

인터랙티브 버전 전에 각 패널이 제대로 그려지는지 확인한다.
한 조합(예: glass, φ=0.5)의 스냅샷을 만든다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import physics as P
import particles as PA
import precompute as PC

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

R = 1e-3


def render_case(sname, theta, phi, data, ax_cross, ax_prof, ax_top, ax_metric,
                traj_frac=1.0):
    """한 조합을 4개 축에 렌더."""
    col = "#BA7517"

    # ================= [1] 단면 + 입자 =================
    ax = ax_cross
    ax.clear()
    r_plot = np.linspace(0, R, 200)
    h = P.droplet_height(r_plot, R, theta)
    # 좌우 대칭 단면
    ax.plot(r_plot / R, h / R, color="#378ADD", lw=1.5)
    ax.plot(-r_plot / R, h / R, color="#378ADD", lw=1.5)
    ax.fill_between(r_plot / R, h / R, alpha=0.10, color="#378ADD")
    ax.fill_between(-r_plot / R, h / R, alpha=0.10, color="#378ADD")

    # 입자 (궤적의 특정 시점)
    traj = data["traj"]
    frame = int((len(traj) - 1) * traj_frac)
    rp = traj[frame]
    # 입자를 좌우로 분산 배치 (시각화)
    signs = np.where(np.arange(len(rp)) % 2 == 0, 1, -1)
    # 입자 높이: 액적 표면 근처 랜덤
    hp = P.droplet_height(rp, R, theta) / R
    yp = np.random.default_rng(0).uniform(0.1, 0.9, len(rp)) * hp
    ax.scatter(signs * rp / R, yp, s=6, color=col, alpha=0.6, zorder=5)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("r / R")
    ax.set_ylabel("h / R")
    ax.set_title(f"[1] Cross-section + particles  (t={traj_frac:.0%})",
                 fontsize=10)
    ax.axhline(0, color='#888', lw=0.5)

    # ================= [2] 농도 프로파일 =================
    ax = ax_prof
    ax.clear()
    centers, density = data["centers"], data["density"]
    ax.fill_between(centers / R, density, alpha=0.35, color=col)
    ax.plot(centers / R, density, color=col, lw=1.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("r / R  (center -> edge)")
    ax.set_ylabel("deposit density")
    ax.set_title("[2] Final deposit profile", fontsize=10)
    ax.grid(alpha=0.3)

    # ================= [3] Top view =================
    ax = ax_top
    ax.clear()
    img = data["topview"]
    im = ax.imshow(img, extent=[-1, 1, -1, 1], origin='lower',
                   cmap='YlOrBr', vmin=0, vmax=1)
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title("[3] Top view (dried stain)", fontsize=10)
    ax.set_aspect('equal')

    # ================= [4] 지표 =================
    ax = ax_metric
    ax.clear()
    ax.axis('off')
    # 막대: 가장자리 vs 중앙 축적률
    bars = ["edge", "center"]
    vals = [data["edge_frac"] * 100, data["center_frac"] * 100]
    bcol = ["#378ADD", "#D85A30"]
    y = [0.62, 0.42]
    for yy, b, v, c in zip(y, bars, vals, bcol):
        ax.barh([yy], [v], height=0.12, color=c, alpha=0.75)
        ax.text(0, yy + 0.10, f"{b}: {v:.0f}%", fontsize=10, transform=ax.transAxes)
        ax.text(min(v, 95) + 2, yy, f"{v:.0f}%", va='center', fontsize=9,
                transform=ax.transAxes)
    ax.set_xlim(0, 100)

    # 텍스트 지표
    ax.text(0.0, 0.95, f"substrate: {sname} (theta={theta}deg)",
            fontsize=10, transform=ax.transAxes, fontweight='bold')
    ax.text(0.0, 0.85, f"ethanol phi = {phi*100:.0f}%",
            fontsize=10, transform=ax.transAxes)
    ax.text(0.0, 0.22, f"Ma = {data['ma']:.1e}",
            fontsize=10, transform=ax.transAxes)
    ptn_col = {"RING": "#378ADD", "CENTER": "#D85A30", "MIXED": "#BA7517"}
    ax.text(0.0, 0.10, f"pattern: {data['pattern']}  "
            f"(edge/center={data['ratio']:.2f})",
            fontsize=11, transform=ax.transAxes, fontweight='bold',
            color=ptn_col[data['pattern']])
    ax.set_title("[4] Metrics", fontsize=10)


if __name__ == "__main__":
    # 캐시 (glass, 네 케이스)
    cache = PC.build_cache(phi_values=[0.0, 0.30, 0.50, 0.70],
                           substrates={"glass": 19.5}, verbose=False)

    # 2x2 조합 스냅샷 4장 (E0, E3, E5, E7)
    for phi in [0.0, 0.30, 0.50, 0.70]:
        data = cache[("glass", round(phi, 2))]
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        fig.suptitle(f"Coffee-ring simulator  |  glass, ethanol {phi*100:.0f}%",
                     fontsize=13, fontweight='bold')
        render_case("glass", 19.5, phi, data, ax1, ax2, ax3, ax4)
        fname = f"layout_E{int(phi*10)}.png"
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fname}")
