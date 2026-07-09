"""
fine_sweep.py — 촘촘한 농도 스윕 (정밀 분석)

실험은 4개 농도(0,30,50,70%)뿐이지만, 시뮬레이션은 촘촘하게 돌릴 수 있다.
2.5% 간격으로 0~70%를 계산하고, 각 농도에서 여러 시드로 반복 평균하여
edge/center 곡선의 전환점을 정밀하게 찾는다.

두 기판(유리, OHP) 모두 계산하여 기판별 최적 농도 차이도 정량화한다.
"""

import numpy as np
from engine import CoffeeRingSim


def measure(phi, theta_deg, n_particles=3000, n_seeds=5):
    """한 농도에서 여러 시드로 반복 → 평균 edge/center, edge%, center%."""
    ratios, edges, centers = [], [], []
    for s in range(n_seeds):
        sim = CoffeeRingSim(phi=phi, theta_deg=theta_deg,
                            n_particles=n_particles, seed=s, auto_loop=False)
        while sim.step():
            pass
        m = sim.metrics()
        ratios.append(m["ratio"])
        edges.append(m["edge_frac"])
        centers.append(m["center_frac"])
    return (np.mean(ratios), np.std(ratios),
            np.mean(edges), np.mean(centers))


def sweep(theta_deg, phi_grid, label):
    print(f"\n{'='*64}")
    print(f"  {label} (theta={theta_deg}deg) — 촘촘한 스윕")
    print(f"{'='*64}")
    print(f"{'phi(%)':>7} {'e/c':>8} {'±std':>7} {'edge%':>7} {'center%':>8}  pattern")
    results = []
    for phi in phi_grid:
        r, rstd, e, c = measure(phi, theta_deg)
        pat = "RING" if r > 1.3 else ("CENTER" if r < 0.77 else "MIXED")
        results.append((phi, r, rstd, e, c, pat))
        print(f"{phi*100:>7.1f} {r:>8.2f} {rstd:>7.2f} {e*100:>7.0f} {c*100:>8.0f}  {pat}")
    return results


def find_transition(results):
    """edge/center = 1을 통과하는 농도를 선형 보간으로 정밀 추정."""
    for i in range(len(results) - 1):
        r1 = results[i][1]
        r2 = results[i + 1][1]
        if (r1 - 1) * (r2 - 1) <= 0 and r1 != r2:   # 1을 사이에 둠
            p1 = results[i][0] * 100
            p2 = results[i + 1][0] * 100
            # 선형 보간: r=1이 되는 phi
            phi_c = p1 + (1 - r1) * (p2 - p1) / (r2 - r1)
            return phi_c
    return None


if __name__ == "__main__":
    phi_grid = np.arange(0, 0.725, 0.025)   # 0~70%, 2.5% 간격

    glass = sweep(19.5, phi_grid, "유리")
    ohp = sweep(50.0, phi_grid, "OHP")

    tg = find_transition(glass)
    to = find_transition(ohp)

    print(f"\n{'='*64}")
    print("  전환점 (edge/center = 1, 링↔중심 균형)")
    print(f"{'='*64}")
    print(f"  유리 (19.5deg): phi_c ≈ {tg:.1f}%" if tg else "  유리: 전환점 범위 밖")
    print(f"  OHP  (50.0deg): phi_c ≈ {to:.1f}%" if to else "  OHP: 전환점 범위 밖")

    # 결과 저장 (차트용)
    np.savez("sweep_data.npz",
             phi=phi_grid,
             glass_ratio=[r[1] for r in glass],
             glass_std=[r[2] for r in glass],
             glass_edge=[r[3] for r in glass],
             glass_center=[r[4] for r in glass],
             ohp_ratio=[r[1] for r in ohp],
             ohp_std=[r[2] for r in ohp],
             ohp_edge=[r[3] for r in ohp],
             ohp_center=[r[4] for r in ohp],
             tg=tg if tg else -1, to=to if to else -1)
    print("\nsaved: sweep_data.npz")
