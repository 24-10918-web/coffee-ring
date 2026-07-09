"""
verify_particles.py — 입자 궤적 + 최종 침착 패턴 검증

개발 2·3단계 통합 검증:
    - 위: φ별 입자 궤적 (r~ vs 정규화 시간) — '과정'
    - 아래: φ별 최종 침착 밀도 프로파일 — '패턴'

링(가장자리 피크)에서 중앙 침착(중심 피크)으로의 전환을 확인한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import physics as P
import particles as PA

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

R = 1e-3
cases = {"E0 (0%)": 0.0, "E3 (30%)": 0.30, "E5 (50%)": 0.50, "E7 (70%)": 0.70}
colors = ["#378ADD", "#1D9E75", "#BA7517", "#D85A30"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Particle Trajectories (top) & Final Deposit Patterns (bottom)  '
             '— glass substrate, no tuning',
             fontsize=14, fontweight='bold')

for j, ((label, phi), col) in enumerate(zip(cases.items(), colors)):
    # --- 궤적 ---
    r_final, r_init, traj, t_rec = PA.track_particles(
        R, 19.5, phi, n_particles=400, seed=1, record_trajectory=True)

    ax = axes[0, j]
    # 궤적: 일부 입자만 (가독성). traj shape [n_rec, n_particles]
    # traj가 t_rec보다 1개 많을 수 있음 (마지막 최종위치). 길이 맞춤.
    m = min(traj.shape[0], len(t_rec))
    traj = traj[:m]
    t_plot = t_rec[:m]
    n_show = 80
    idx = np.linspace(0, traj.shape[1] - 1, n_show).astype(int)
    for p in idx:
        ax.plot(traj[:, p] / R, t_plot, color=col, alpha=0.25, lw=0.6)
    ax.set_title(f"{label}", fontsize=11, fontweight='bold')
    ax.set_xlabel("r / R")
    ax.set_ylabel("normalized time" if j == 0 else "")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()   # 시간 위→아래
    ax.grid(alpha=0.3)

    # --- 최종 침착 패턴 ---
    r_big, r_init_b = PA.track_particles(R, 19.5, phi, n_particles=4000, seed=2)
    centers, density, counts = PA.deposit_histogram(r_big, R, n_bins=35)

    axd = axes[1, j]
    axd.fill_between(centers / R, density, color=col, alpha=0.35)
    axd.plot(centers / R, density, color=col, lw=1.8)
    axd.set_xlabel("r / R")
    axd.set_ylabel("deposit density" if j == 0 else "")
    axd.set_xlim(0, 1)
    axd.grid(alpha=0.3)

    # 패턴 라벨
    edge_val = np.mean(density[int(0.80 * len(density)):])
    center_val = np.mean(density[:int(0.20 * len(density))])
    ratio = edge_val / (center_val + 1e-9)
    ptn = "RING" if ratio > 1.3 else ("CENTER" if ratio < 0.77 else "MIXED")
    axd.text(0.5, 0.92, f"{ptn}\n(edge/center={ratio:.2f})",
             transform=axd.transAxes, ha='center', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.savefig('particle_verification.png', dpi=130, bbox_inches='tight')
print("Saved: particle_verification.png")

# 요약 출력
print("\n패턴 전환 요약:")
for label, phi in cases.items():
    r_big, _ = PA.track_particles(R, 19.5, phi, n_particles=4000, seed=2)
    centers, density, _ = PA.deposit_histogram(r_big, R, n_bins=35)
    edge_val = np.mean(density[int(0.80 * len(density)):])
    center_val = np.mean(density[:int(0.20 * len(density))])
    ratio = edge_val / (center_val + 1e-9)
    print(f"  {label}: edge/center = {ratio:.2f}")
