"""
particles.py — 입자 궤적 추적 (라그랑주 방식)

속도장 u_net 위에 입자 N개를 뿌리고 RK4로 시간 적분한다.
각 입자의 반경 위치 r_i(t)를 추적하여 '과정'을 시각화하고,
최종 분포 히스토그램이 침착 패턴이 된다.

핵심 물리:
    dr_i/dt = u_net(r_i)  (반경 방향 이동)
    - 물이 증발하면서 액적이 얇아지고, 입자가 국소 높이보다
      두꺼워지면(=바닥에 닿으면) 그 자리에 고정(pinning)된다.
    - 가장자리 도달 입자도 고정 (접촉선 pinning).

이 방식은 속도장을 직접 따라가므로 물리가 투명하고 검증이 쉽다.
"""

import numpy as np
import physics as P


def track_particles(R, theta_deg, phi, n_particles=500, n_steps=1500,
                    seed=0, record_trajectory=False):
    """입자 궤적 적분 (무차원화).

    무차원화:
        길이: r~ = r/R  (0~1)
        시간: t~ = t/t_dry, 건조시간 t_dry로 정규화
        속도: 모세관·마랑고니를 같은 기준으로 스케일

    이렇게 하면 절대 단위가 사라지고 두 흐름의 '비율'만 물리를 지배한다.
    가장자리 이동과 증발이 같은 시간 스케일에서 경쟁한다.

    Returns:
        r_final, r_init [, traj, t_rec]
    """
    rng = np.random.default_rng(seed)

    # 무차원 초기 배치 (면적 균일): r~ in [0,1]
    u = rng.uniform(0, 1, n_particles)
    rn = np.sqrt(u)                # 무차원 반경
    r_init = rn.copy() * R
    active = np.ones(n_particles, dtype=bool)

    theta = np.radians(theta_deg)
    lam = P.lambda_exponent(theta_deg)

    # --- 무차원 속도장 함수 (r~ 입력, 0~1) ---
    # 모세관: 증발 보충 흐름. u_cap~ = (1/(r~·H~)) ∫ J~ r~ dr~
    # 여기서 형상 H~(r~)=1-r~², 증발 J~(r~)=(1-r~²)^(-λ)
    # 마랑고니: 표면장력 구배 기반, 세기는 φ에 의존.
    def vel_nondim(rr):
        rr = np.clip(rr, 1e-5, 1 - 1e-5)
        Hn = 1 - rr ** 2                       # 무차원 높이
        # 모세관 (해석적 근사): 가장자리로 갈수록 급증
        # ∫₀ʳ (1-s²)^(-λ) s ds 를 수치적으로 누적하는 대신,
        # 국소 형태로 근사: u_cap~ ∝ [ (1-r²)^(1-λ) 관련 ] / (r·H)
        # 안정적 형태 사용:
        u_cap = rr * (1 - rr ** 2) ** (-lam) / (Hn + 1e-6)
        u_cap = u_cap * 0.15                   # 스케일 상수 (무차원)

        # 마랑고니 (안쪽, 음수). 표면장력 구배로 생기는 표면 역류.
        # 모세관과 달리 가장자리 발산이 아니라, 표면층 전체에 작용하는
        # 비교적 완만한 안쪽 흐름. 세기는 표면장력 낙폭(φ 의존)에 비례.
        dg = abs(P.surface_tension(0.0) - P.surface_tension(phi))  # 총 낙폭
        mar_strength = dg / P.surface_tension(0.0)   # 0~0.7
        mu_rel = P.viscosity(phi) / P.viscosity(0.0)  # 점성 비 (역류 억제)
        # 반경 프로파일: r(1-r²) 형태 — 중간에서 최대, 양 끝 0.
        # 안쪽으로 미는 완만한 흐름.
        u_mar = -rr * (1 - rr ** 2) * (mar_strength / mu_rel) * 2.2

        return u_cap + u_mar

    # --- 무차원 시간 ---
    # 건조 시간을 1로 정규화. 증발이 빠른 고농도(φ↑)는 t_dry가 짧음.
    # 상대 건조 속도: J0(φ) 클수록 빨리 마름 → 유효 시간 짧음.
    evap_speed = P.evaporation_rate(phi)          # 물=1, 에탄올=2.5
    t_dry = 1.0 / evap_speed                       # 무차원 건조 시간
    dt = t_dry / n_steps

    # 높이 스케일 감소 (무차원): H_scale 1 → 0
    H_scale = 1.0
    dH = -1.0 / n_steps                            # 선형 감소로 t_dry에 0 도달

    rec_every = max(1, n_steps // 60)
    traj = [] if record_trajectory else None
    t_rec = [] if record_trajectory else None

    for step in range(n_steps):
        H_scale = max(H_scale + dH, 0.0)

        if np.any(active):
            ra = rn[active]
            k1 = vel_nondim(ra)
            k2 = vel_nondim(np.clip(ra + 0.5 * dt * k1, 1e-5, 1))
            k3 = vel_nondim(np.clip(ra + 0.5 * dt * k2, 1e-5, 1))
            k4 = vel_nondim(np.clip(ra + dt * k3, 1e-5, 1))
            ra_new = np.clip(ra + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4),
                             1e-5, 1.0)
            rn[active] = ra_new

            # 고정 조건: 가장자리 도달 OR 국소 높이 소진
            reached_edge = rn >= 1 - 1e-4
            local_H = H_scale * (1 - rn ** 2)      # 무차원 국소 높이
            too_thin = local_H <= 0.02
            newly = active & (reached_edge | too_thin)
            active[newly] = False

        if record_trajectory and step % rec_every == 0:
            traj.append(rn.copy() * R)
            t_rec.append(step * dt / t_dry)        # 0~1 정규화 시간

        if not np.any(active) or H_scale <= 0.02:
            break

    r_final = rn.copy() * R

    if record_trajectory:
        traj.append(r_final.copy())
        return r_final, r_init, np.array(traj), np.array(t_rec)
    return r_final, r_init


def deposit_histogram(r_final, R, n_bins=40):
    """최종 입자 위치 → 반경별 밀도 (면적 정규화).

    각 링 영역의 면적으로 나눠 단위 면적당 밀도를 구한다.
    """
    edges = np.linspace(0, R, n_bins + 1)
    counts, _ = np.histogram(r_final, bins=edges)
    # 각 bin의 면적 (환형): π(r_out² - r_in²)
    ring_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density = counts / (ring_area + 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density, counts


if __name__ == "__main__":
    print("=" * 60)
    print("particles.py 검증 — 입자 최종 분포 패턴")
    print("=" * 60)
    R = 1e-3
    cases = {"E0": 0.0, "E3": 0.30, "E5": 0.50, "E7": 0.70}

    for label, phi in cases.items():
        r_final, r_init = track_particles(R, 19.5, phi, n_particles=2000, seed=1)
        centers, density, counts = deposit_histogram(r_final, R, n_bins=30)

        # 패턴 분석
        edge_zone = density[int(0.80 * len(density)):]
        center_zone = density[:int(0.20 * len(density))]
        edge_val = np.mean(edge_zone)
        center_val = np.mean(center_zone)
        ratio = edge_val / (center_val + 1e-9)

        # 가장자리 도달 비율
        frac_edge = np.mean(r_final >= R - 1e-4) * 100

        pattern = "링 우세" if ratio > 1.5 else \
                  ("중앙 우세" if ratio < 0.67 else "혼재")
        print(f"\n{label} (phi={phi:.2f}):")
        print(f"  가장자리 도달 입자: {frac_edge:.0f}%")
        print(f"  edge/center 밀도비: {ratio:.2f} → {pattern}")
