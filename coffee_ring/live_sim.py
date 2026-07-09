"""
live_sim.py — 실시간 입자 동역학 시뮬레이터 (개선판)

개선 사항:
    1. 입자 색을 '시작 위치'로 고정 (움직여도 안 바뀜)
       → 안쪽 출발(파랑)/바깥 출발(빨강) 입자를 끝까지 추적 가능.
    2. 자동 재생(auto-play) — 열면 바로 입자가 흐른다.
    3. 마르면 자동 리셋(loop) — 건조 과정을 반복 재생.
    4. 마랑고니 강화로 φ별 차이 뚜렷 (E0 링 ↔ E7 중앙).

레이아웃:
    [메인 좌] 위에서 본 2D 평면 — 입자 실시간 유동
    [우 상]   단면 (r-z)
    [우 중]   반경 밀도 프로파일
    [우 하]   지표

실행: python live_sim.py  (로컬 GUI 필요)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.animation as animation
import physics as P
from engine import CoffeeRingSim

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

SUBSTRATES = {"glass": 19.5, "OHP": 50.0}

# 엔진 (자동 루프 ON)
sim = CoffeeRingSim(R=1.0, theta_deg=19.5, phi=0.50,
                    n_particles=1500, seed=0, auto_loop=False)

# ============================================================
# UI 레이아웃
# ============================================================
fig = plt.figure(figsize=(15, 9))
fig.suptitle("Live Coffee-ring Simulator — real-time drying dynamics "
             "(theory-based, no tuning)", fontsize=13, fontweight='bold')

gs = GridSpec(3, 2, figure=fig, left=0.06, right=0.97,
              top=0.90, bottom=0.20, hspace=0.45, wspace=0.22,
              width_ratios=[1.4, 1.0])

ax_main = fig.add_subplot(gs[:, 0])
ax_cross = fig.add_subplot(gs[0, 1])
ax_prof = fig.add_subplot(gs[1, 1])
ax_metric = fig.add_subplot(gs[2, 1])

# 컨트롤
ax_slider = fig.add_axes([0.12, 0.10, 0.55, 0.03])
phi_slider = Slider(ax_slider, "ethanol phi (%)", 0, 70, valinit=50, valstep=5)

ax_radio = fig.add_axes([0.75, 0.05, 0.13, 0.09])
radio = RadioButtons(ax_radio, ("glass", "OHP"), active=0)

ax_play = fig.add_axes([0.12, 0.04, 0.10, 0.04])
btn_play = Button(ax_play, "Pause")   # 시작이 재생 상태
ax_reset = fig.add_axes([0.24, 0.04, 0.10, 0.04])
btn_reset = Button(ax_reset, "Restart")
ax_save = fig.add_axes([0.36, 0.04, 0.10, 0.04])
btn_save = Button(ax_save, "Save PNG")

state = {"playing": True}   # ★ 자동 재생으로 시작

# ---- 메인: top view scatter (색 = 시작 위치, 고정) ----
main_scat = ax_main.scatter(sim.x, sim.y, c=sim.color0,
                            cmap='coolwarm', s=9, alpha=0.75,
                            vmin=0, vmax=1)
edge_circle = plt.Circle((0, 0), 1.0, fill=False, color='#888',
                         lw=1.2, ls='--')
ax_main.add_patch(edge_circle)
ax_main.set_xlim(-1.15, 1.15)
ax_main.set_ylim(-1.15, 1.15)
ax_main.set_aspect('equal')
ax_main.set_title("Top view — color = starting position (fixed)",
                  fontsize=11)
ax_main.set_xlabel("x / R")
ax_main.set_ylabel("y / R")
info_text = ax_main.text(0.02, 0.98, "", transform=ax_main.transAxes,
                         fontsize=10, va='top',
                         bbox=dict(boxstyle='round', facecolor='white',
                                   alpha=0.85))
# 색 설명
ax_main.text(0.98, 0.02, "blue = started inner\nred = started outer",
             transform=ax_main.transAxes, fontsize=8, va='bottom', ha='right',
             color='#555',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.7))


def draw_cross():
    ax_cross.clear()
    r_plot = np.linspace(0, 1, 200)
    h = (1 - r_plot**2) * sim.H_scale * np.tan(np.radians(sim.theta_deg)/2)
    ax_cross.plot(r_plot, h, color="#378ADD", lw=1.3)
    ax_cross.plot(-r_plot, h, color="#378ADD", lw=1.3)
    ax_cross.fill_between(r_plot, h, alpha=0.10, color="#378ADD")
    ax_cross.fill_between(-r_plot, h, alpha=0.10, color="#378ADD")
    r = np.sqrt(sim.x**2 + sim.y**2)
    sign = np.sign(sim.x + 1e-9)
    hh = np.maximum((1 - r**2) * sim.H_scale
                    * np.tan(np.radians(sim.theta_deg)/2), 0.001)
    yp = np.random.default_rng(0).uniform(0.05, 0.95, len(r)) * hh
    ax_cross.scatter(sign * r, yp, s=4, c=sim.color0, cmap='coolwarm',
                     vmin=0, vmax=1, alpha=0.6)
    ax_cross.set_xlim(-1.1, 1.1)
    # y축 상한을 접촉각 기준 최대 액적 높이에 맞춰 자동 조정.
    # (고정값이면 OHP처럼 접촉각 큰 기판에서 액적이 잘림)
    h_max = np.tan(np.radians(sim.theta_deg) / 2)   # 초기 중심 높이
    ax_cross.set_ylim(0, h_max * 1.2 + 0.02)
    ax_cross.set_title("Cross-section (drying)", fontsize=10)
    ax_cross.set_xlabel("r / R")


def draw_prof():
    ax_prof.clear()
    centers, density = sim.radial_density(n_bins=40)
    ax_prof.fill_between(centers, density, alpha=0.35, color="#BA7517")
    ax_prof.plot(centers, density, color="#BA7517", lw=1.6)
    ax_prof.set_xlim(0, 1)
    ax_prof.set_title("Radial density (live)", fontsize=10)
    ax_prof.set_xlabel("r / R")
    ax_prof.set_ylabel("density")
    ax_prof.grid(alpha=0.3)


def draw_metric():
    ax_metric.clear()
    ax_metric.axis('off')
    m = sim.metrics()
    ax_metric.text(0.0, 0.90, f"drying progress: {m['progress']*100:.0f}%",
                   fontsize=10, transform=ax_metric.transAxes)
    ax_metric.text(0.0, 0.72, f"edge: {m['edge_frac']*100:.0f}%   "
                   f"center: {m['center_frac']*100:.0f}%",
                   fontsize=10, transform=ax_metric.transAxes)
    ax_metric.text(0.0, 0.54, f"Ma = {m['ma']:.1e}",
                   fontsize=10, transform=ax_metric.transAxes)
    pcol = {"RING": "#378ADD", "CENTER": "#D85A30", "MIXED": "#BA7517"}
    ax_metric.text(0.0, 0.30, f"pattern: {m['pattern']}",
                   fontsize=13, fontweight='bold', color=pcol[m['pattern']],
                   transform=ax_metric.transAxes)
    ax_metric.text(0.0, 0.14, f"edge/center = {m['ratio']:.2f}",
                   fontsize=10, transform=ax_metric.transAxes)
    ax_metric.set_title("Metrics (live)", fontsize=10)


def refresh_side():
    draw_cross()
    draw_prof()
    draw_metric()


# ============================================================
# 애니메이션 루프
# ============================================================
def update(frame):
    if state["playing"]:
        for _ in range(3):     # 프레임당 여러 스텝 (부드럽게)
            alive = sim.step()
            if not alive:       # 건조 완료 (auto_loop=False)
                break
        main_scat.set_offsets(np.c_[sim.x, sim.y])
        # 색은 고정(color0) — set_array로 다시 지정해도 동일값이라 안 바뀜
        main_scat.set_array(sim.color0)
        m = sim.metrics()
        info_text.set_text(
            f"phi={sim.phi*100:.0f}%  {sim.theta_deg:.0f}deg\n"
            f"drying {m['progress']*100:.0f}%  |  {m['pattern']}")
        refresh_side()

        # 건조 완료 시 자동 정지 (100%에서 멈춤, Restart로 재시작)
        if sim.is_done():
            state["playing"] = False
            btn_play.label.set_text("Play")
            info_text.set_text(
                f"phi={sim.phi*100:.0f}%  {sim.theta_deg:.0f}deg\n"
                f"DONE (100%)  |  {m['pattern']}  —  press Restart")
    return [main_scat]


ani = animation.FuncAnimation(fig, update, interval=50, blit=False,
                              cache_frame_data=False)


# ============================================================
# 콜백
# ============================================================
def apply_reset():
    main_scat.set_offsets(np.c_[sim.x, sim.y])
    main_scat.set_array(sim.color0)
    # scatter 색범위 재설정 (새 color0)
    main_scat.set_clim(0, 1)
    refresh_side()
    fig.canvas.draw_idle()


def on_phi(val):
    sim.set_params(phi=val / 100.0)
    apply_reset()


def on_radio(label):
    sim.set_params(theta_deg=SUBSTRATES[label])
    apply_reset()


def on_play(event):
    # 이미 다 마른 상태에서 Play를 누르면 새로 시작
    if sim.is_done() and not state["playing"]:
        sim.reset()
        apply_reset()
    state["playing"] = not state["playing"]
    btn_play.label.set_text("Play" if not state["playing"] else "Pause")
    fig.canvas.draw_idle()


def on_reset(event):
    sim.reset()
    apply_reset()


def on_save(event):
    fname = f"live_{radio.value_selected}_E{int(sim.phi*10)}.png"
    fig.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"Saved: {fname}")


phi_slider.on_changed(on_phi)
radio.on_clicked(on_radio)
btn_play.on_clicked(on_play)
btn_reset.on_clicked(on_reset)
btn_save.on_clicked(on_save)

refresh_side()

if __name__ == "__main__":
    plt.show()
