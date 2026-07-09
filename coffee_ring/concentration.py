"""
concentration.py — 농도장 솔버 (오일러 방식)

수정된 깊이 평균 이류 방정식을 유한체적법(FVM)으로 푼다:

    d(h·c)/dt + (1/r)·d(r·h·u_net·c)/dr = 0

핵심: h(r,t)가 증발로 감소(dh/dt = -J/rho)하므로,
유속이 0이어도 남은 용질이 농축된다. 이 항이 링/중심침착 재현의 열쇠.

경계 조건:
    - r=0 (중심): 대칭 (플럭스 0)
    - r=R (가장자리): 접촉선 고정, 용질은 빠져나가지 못함 (플럭스 0)
      → 용질 총량 보존 (증발하는 건 용매인 물뿐)
"""

import numpy as np
import physics as P


def solve_concentration(R, theta_deg, phi, n_cells=200, n_steps=4000,
                        dry_fraction=0.97, verbose=False):
    """농도장 시간 발전.

    Returns:
        r_centers: 셀 중심 반경 좌표
        c_final: 최종 용질 농도 프로파일 (깊이 적분 밀도 h·c에 해당)
        surface_density: 단위 면적당 최종 침착량 (마른 뒤 관측량)
        mass_history: 시간별 총 용질량 (보존 확인용)
    """
    # 반경 격자 (셀 중심)
    dr = R / n_cells
    r_edges = np.linspace(0, R, n_cells + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # 초기 액적 높이
    h = P.droplet_height(r_centers, R, theta_deg)
    h = np.maximum(h, 1e-9)

    # 초기 용질 농도 c (균일). 관측량은 h*c (깊이 적분).
    c = np.ones(n_cells)

    # 속도장 (셀 경계에서 평가) — 시간 동안 형상 비율 유지 가정
    u_net_c, _, _ = P.net_velocity(r_centers, R, theta_deg, phi)
    # 경계 속도: 인접 셀 평균
    u_edges = np.zeros(n_cells + 1)
    u_edges[1:-1] = 0.5 * (u_net_c[:-1] + u_net_c[1:])
    u_edges[0] = 0.0       # 중심 대칭
    u_edges[-1] = 0.0      # 가장자리 고정 (용질 유출 없음)

    # 증발에 의한 높이 감소율 dh/dt = -J/rho (상대)
    J = P.evaporation_flux(r_centers, R, theta_deg, phi)
    # 정규화: 중심 증발률 기준 상대 시간
    J_norm = J / np.mean(J)

    # 초기 총 용질량 (∫ h·c·r dr, 축대칭 부피 가중)
    def total_mass(h_, c_):
        return np.sum(h_ * c_ * r_centers * dr)

    m0 = total_mass(h, c)
    mass_history = [1.0]

    # 시간 스텝
    h_mean0 = np.mean(h)
    total_dry_time = dry_fraction * np.max(h)  # 최대 높이 기준 (완전 건조)
    dt = total_dry_time / n_steps

    # 침착 누적: 각 셀에서 '말라붙은' 용질량을 적산.
    # 물리: 국소 높이 h가 임계 이하로 얇아지면 그 위치 용질은 고정(침착)된다.
    deposit = np.zeros(n_cells)
    h_pin = 0.02 * np.max(h)   # 이 두께 이하로 얇아지면 침착 시작

    for step in range(n_steps):
        hc = h * c

        # --- 이류 플럭스 (1차 풍상차분, upwind) ---
        F = np.zeros(n_cells + 1)
        pos = u_edges > 0
        # 바깥 방향: 왼쪽(안쪽) 셀 값 사용
        F[1:-1][pos[1:-1]] = (r_edges[1:-1] * u_edges[1:-1] * hc[:-1])[pos[1:-1]]
        # 안쪽 방향: 오른쪽(바깥) 셀 값 사용
        F[1:-1][~pos[1:-1]] = (r_edges[1:-1] * u_edges[1:-1] * hc[1:])[~pos[1:-1]]

        dhc_adv = -(F[1:] - F[:-1]) / (r_centers * dr)
        hc_new = np.maximum(hc + dt * dhc_adv, 0.0)

        # --- 증발: 높이 감소 (용매만) ---
        h = np.maximum(h - dt * J_norm, 1e-6)

        # 새 농도
        c = hc_new / h

        # --- 침착: 얇아진 곳의 용질을 고정시켜 deposit에 이관 ---
        pinned = h <= h_pin
        if np.any(pinned):
            deposit[pinned] += (h[pinned] * c[pinned])
            c[pinned] = 0.0           # 침착된 용질은 유동에서 제거
            h[pinned] = 1e-6

        mass_history.append((total_mass(h, c) + np.sum(deposit * r_centers * dr)) / m0)

        if np.max(h) <= 1.5 * h_pin:
            # 남은 용질 전부 침착
            deposit += h * c
            break

    # 최종 침착 = 각 위치에 고정된 용질량 밀도
    surface_density = deposit.copy()

    return r_centers, deposit, surface_density, np.array(mass_history)


if __name__ == "__main__":
    print("=" * 60)
    print("concentration.py 검증 — 질량 보존 & 패턴")
    print("=" * 60)
    R = 1e-3
    cases = {"E0": 0.0, "E3": 0.30, "E5": 0.50, "E7": 0.70}

    for label, phi in cases.items():
        r, dep, sd, mass = solve_concentration(R, 19.5, phi, verbose=False)
        # 패턴 분석
        edge_zone = dep[int(0.85 * len(dep)):]      # 바깥 15%
        center_zone = dep[:int(0.15 * len(dep))]     # 안쪽 15%
        edge_peak = np.max(edge_zone)
        center_peak = np.max(center_zone)
        mass_drift = abs(mass[-1] - 1.0) * 100
        print(f"\n{label} (phi={phi:.2f}):")
        print(f"  질량 보존 오차: {mass_drift:.2f}%")
        print(f"  가장자리 축적: {edge_peak:.3f} / 중앙 축적: {center_peak:.3f}")
        ratio = edge_peak / (center_peak + 1e-9)
        pattern = "링 우세" if ratio > 1.3 else ("중앙 우세" if ratio < 0.77 else "혼재")
        print(f"  edge/center 비: {ratio:.2f} → {pattern}")
