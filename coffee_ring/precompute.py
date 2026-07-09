"""
precompute.py — φ·기판 조합별 시뮬레이션 사전 계산

슬라이더를 움직일 때마다 전체 시뮬레이션을 다시 돌리면 느리므로,
가능한 조합을 미리 계산해 캐시한다. 인터랙티브 화면은 캐시를 전환만 한다.

각 조합에 대해 저장:
    - 입자 최종 위치 (r_final)
    - 궤적 (traj, t_rec) — 애니메이션용
    - 침착 밀도 프로파일 (centers, density)
    - top view 2D 이미지 (극좌표 → 직교)
    - 지표 (Ma, 가장자리/중앙 축적률)
"""

import numpy as np
import physics as P
import particles as PA


def compute_case(R, theta_deg, phi, n_particles=4000, seed=2):
    """한 조합의 모든 결과 계산."""
    # 최종 위치 (많은 입자로 패턴 통계)
    r_final, r_init = PA.track_particles(R, theta_deg, phi,
                                         n_particles=n_particles, seed=seed)
    centers, density, counts = PA.deposit_histogram(r_final, R, n_bins=40)

    # 궤적 (적은 입자로 애니메이션)
    _, _, traj, t_rec = PA.track_particles(R, theta_deg, phi,
                                           n_particles=200, seed=1,
                                           record_trajectory=True)

    # 지표
    edge_val = np.mean(density[int(0.80 * len(density)):])
    center_val = np.mean(density[:int(0.20 * len(density))])
    total = np.sum(counts)
    edge_frac = np.sum(counts[int(0.80 * len(counts)):]) / (total + 1e-9)
    center_frac = np.sum(counts[:int(0.20 * len(counts))]) / (total + 1e-9)
    ma = P.marangoni_number(R, theta_deg, phi)
    ratio = edge_val / (center_val + 1e-9)
    if ratio > 1.3:
        pattern = "RING"
    elif ratio < 0.77:
        pattern = "CENTER"
    else:
        pattern = "MIXED"

    # top view 이미지 (density를 극좌표 회전)
    topview = make_topview(centers, density, R, size=200)

    return {
        "phi": phi, "theta": theta_deg,
        "r_final": r_final, "centers": centers, "density": density,
        "traj": traj, "t_rec": t_rec,
        "edge_frac": edge_frac, "center_frac": center_frac,
        "ma": ma, "ratio": ratio, "pattern": pattern,
        "topview": topview,
    }


def make_topview(centers, density, R, size=200):
    """반경 밀도 → 위에서 본 2D 원형 얼룩 이미지."""
    x = np.linspace(-R, R, size)
    y = np.linspace(-R, R, size)
    X, Y = np.meshgrid(x, y)
    rr = np.sqrt(X ** 2 + Y ** 2)
    # 반경별 밀도를 보간
    dens_norm = density / (np.max(density) + 1e-12)
    img = np.interp(rr.ravel(), centers, dens_norm,
                    left=dens_norm[0], right=0).reshape(rr.shape)
    img[rr > R] = np.nan   # 액적 밖은 투명
    return img


def build_cache(R=1e-3, phi_values=None, substrates=None,
                n_particles=4000, verbose=True):
    """전체 캐시 생성.

    phi_values: 계산할 에탄올 농도 리스트 (부피분율)
    substrates: {이름: 접촉각(도)}
    """
    if phi_values is None:
        phi_values = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    if substrates is None:
        substrates = {"glass": 19.5, "OHP": 50.0}

    cache = {}
    for sname, theta in substrates.items():
        for phi in phi_values:
            key = (sname, round(phi, 2))
            if verbose:
                print(f"  computing {sname} phi={phi:.2f} ...", flush=True)
            cache[key] = compute_case(R, theta, phi, n_particles=n_particles)
    if verbose:
        print(f"cache built: {len(cache)} cases")
    return cache


if __name__ == "__main__":
    print("=" * 60)
    print("precompute.py — 캐시 생성 테스트")
    print("=" * 60)
    cache = build_cache(phi_values=[0.0, 0.30, 0.50, 0.70],
                        substrates={"glass": 19.5})
    print("\n결과 요약:")
    for key, d in cache.items():
        print(f"  {key}: pattern={d['pattern']}, ratio={d['ratio']:.2f}, "
              f"edge={d['edge_frac']*100:.0f}% center={d['center_frac']*100:.0f}%")
