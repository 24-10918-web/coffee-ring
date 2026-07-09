"""
coffee_ring_sim.py — 커피링 · 중심 침착 인터랙티브 시뮬레이터 (메인)

실행:
    python coffee_ring_sim.py

조작:
    - 에탄올 슬라이더 (0~70%): 농도 변경 → 패턴 즉시 갱신
    - 기판 라디오 (glass / OHP): 접촉각 변경
    - Play 버튼: 입자 궤적 애니메이션 재생
    - Reset 버튼: 애니메이션 초기화
    - Save 버튼: 현재 화면 PNG 저장

설계 원칙: 모든 파라미터 이론·문헌값, 튜닝 없음.
성능: φ·기판 조합을 사전 계산(캐시)하여 슬라이더는 전환만 수행.

주의: 이 파일은 GUI 환경(로컬 PC)에서 실행해야 슬라이더/애니메이션이
      동작합니다. 서버/노트북 백엔드에서는 정적 렌더만 됩니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, RadioButtons, Button
import physics as P
import particles as PA
import precompute as PC
import render_static as RS

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

R = 1e-3

# ============================================================
# 사전 계산 (캐시)
# ============================================================
print("Precomputing cases (this runs once)...")
PHI_LIST = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
SUBSTRATES = {"glass": 19.5, "OHP": 50.0}
CACHE = PC.build_cache(phi_values=PHI_LIST, substrates=SUBSTRATES,
                       n_particles=4000, verbose=False)
print("Done. Launching UI...")


def nearest_phi(phi):
    """슬라이더 값 → 캐시에 있는 가장 가까운 φ."""
    return min(PHI_LIST, key=lambda p: abs(p - phi))


# ============================================================
# UI 구성
# ============================================================
fig = plt.figure(figsize=(15, 9))
fig.suptitle("Coffee-ring / Center-deposit Simulator  "
             "(theory-based, no tuning)", fontsize=14, fontweight='bold')

gs = GridSpec(2, 2, figure=fig, left=0.07, right=0.97,
              top=0.90, bottom=0.24, hspace=0.35, wspace=0.25)
ax_cross = fig.add_subplot(gs[0, 0])
ax_prof = fig.add_subplot(gs[0, 1])
ax_top = fig.add_subplot(gs[1, 0])
ax_metric = fig.add_subplot(gs[1, 1])

# 컨트롤 영역
ax_slider = fig.add_axes([0.15, 0.13, 0.60, 0.03])
phi_slider = Slider(ax_slider, "ethanol phi (%)", 0, 70,
                    valinit=50, valstep=10)

ax_radio = fig.add_axes([0.80, 0.08, 0.15, 0.10])
radio = RadioButtons(ax_radio, ("glass", "OHP"), active=0)

ax_play = fig.add_axes([0.15, 0.05, 0.10, 0.04])
btn_play = Button(ax_play, "Play")
ax_reset = fig.add_axes([0.27, 0.05, 0.10, 0.04])
btn_reset = Button(ax_reset, "Reset")
ax_save = fig.add_axes([0.39, 0.05, 0.10, 0.04])
btn_save = Button(ax_save, "Save PNG")

# 상태
state = {"sname": "glass", "phi": 0.50, "traj_frac": 1.0,
         "anim": None, "playing": False}


def redraw():
    sname = state["sname"]
    theta = SUBSTRATES[sname]
    phi = nearest_phi(state["phi"])
    data = CACHE[(sname, round(phi, 2))]
    RS.render_case(sname, theta, phi, data,
                   ax_cross, ax_prof, ax_top, ax_metric,
                   traj_frac=state["traj_frac"])
    fig.canvas.draw_idle()


def on_phi(val):
    state["phi"] = val / 100.0
    state["traj_frac"] = 1.0
    redraw()


def on_radio(label):
    state["sname"] = label
    state["traj_frac"] = 1.0
    redraw()


def on_play(event):
    """궤적 애니메이션: traj_frac을 0→1로 증가."""
    import matplotlib.animation as animation
    sname = state["sname"]
    phi = nearest_phi(state["phi"])
    data = CACHE[(sname, round(phi, 2))]
    n_frames = len(data["traj"])

    def update(frame):
        state["traj_frac"] = frame / (n_frames - 1)
        redraw()
        return []

    state["anim"] = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=80, repeat=False, blit=False)
    fig.canvas.draw_idle()


def on_reset(event):
    state["traj_frac"] = 0.0
    redraw()


def on_save(event):
    sname = state["sname"]
    phi = int(nearest_phi(state["phi"]) * 100)
    fname = f"snapshot_{sname}_E{phi//10}.png"
    fig.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"Saved: {fname}")


phi_slider.on_changed(on_phi)
radio.on_clicked(on_radio)
btn_play.on_clicked(on_play)
btn_reset.on_clicked(on_reset)
btn_save.on_clicked(on_save)

# 초기 렌더
redraw()

if __name__ == "__main__":
    plt.show()
