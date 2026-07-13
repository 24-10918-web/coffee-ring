"""
physics.py — 커피링 · 중심 침착 시뮬레이션의 물리 함수 모듈  (v2)

v2 변경점:
    [추가] preferential_depletion(phi) — 라울 법칙에서 유도한 가장자리 에탄올 고갈량
    [추가] marangoni_strength(phi)    — 국소 기울기 기반 마랑고니 세기 (유도)
    [추가] raoult_step()              — 건조 중 조성 phi(t) 변화 (2성분 증발)
    [설계] phi를 상수가 아니라 시간의 함수로 다룬다. 이것이 v1의 핵심 결함이었다.

설계 원칙:
    - gamma(phi), mu(phi), alpha 는 모두 문헌값.
    - 마랑고니 세기의 phi 의존 '형태'는 유도된 것이며 튜닝하지 않았다.
    - 자유 상수는 EPS_DEPLETION 하나뿐이며, 민감도 분석 대상이다.

참고: Deegan et al. (1997), Hu & Larson (2006), Vazquez et al. (1995).
"""

import numpy as np
from scipy.interpolate import interp1d

GAMMA_WATER = 0.072
GAMMA_ETHANOL = 0.022
MU_WATER = 1.00e-3
MU_ETHANOL = 1.20e-3
RHO = 1.0e3
ALPHA = 2.5              # 에탄올/물 상대 휘발도 (v1의 EVAP_RATIO와 동일)

# --- 유일한 자유 상수 ---
# 가장자리에서 증발이 만든 조성 결손이, 확산/이류 재공급에 씻겨나가기 전까지
# 얼마나 축적되는가를 나타내는 O(1) 효율 인자 (Peclet 성격).
# ★ 이 상수는 마랑고니의 '세기'만 정하고 'phi 의존 형태'는 바꾸지 않는다.
EPS_DEPLETION = 1.0

_PHI_DATA = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40,
                      0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
_GAMMA_DATA = np.array([0.0720, 0.0560, 0.0480, 0.0400, 0.0350, 0.0310,
                        0.0285, 0.0265, 0.0250, 0.0238, 0.0228, 0.0220])

_gamma_interp = interp1d(_PHI_DATA, _GAMMA_DATA, kind='cubic',
                         bounds_error=False,
                         fill_value=(_GAMMA_DATA[0], _GAMMA_DATA[-1]))


def surface_tension(phi):
    phi = np.clip(phi, 0.0, 1.0)
    return float(_gamma_interp(phi)) if np.isscalar(phi) else _gamma_interp(phi)


def dgamma_dphi(phi, h=1e-4):
    phi = np.clip(phi, h, 1.0 - h)
    return (surface_tension(phi + h) - surface_tension(phi - h)) / (2 * h)


def viscosity(phi):
    phi = np.clip(phi, 0.0, 1.0)
    mu_linear = MU_WATER * (1 - phi) + MU_ETHANOL * phi
    peak = 1.0 + 1.4 * phi * (1 - phi) * 4.0 * np.exp(-((phi - 0.4) ** 2) / 0.08)
    return mu_linear * peak


def evaporation_rate(phi):
    """J0(phi) = (1-phi)*1 + phi*alpha  (물=1로 정규화). 라울 법칙."""
    phi = np.clip(phi, 0.0, 1.0)
    return (1.0 - phi) + ALPHA * phi


def preferential_depletion(phi, eps=None):
    """가장자리 에탄올 결손 dphi_edge(phi) — 라울 법칙에서 유도.

    라울:  phi_vapor = alpha*phi / (alpha*phi + (1-phi))
    초과분: phi_vapor - phi = (alpha-1)*phi*(1-phi) / (1 + (alpha-1)*phi)

    ★ phi->0 에서 0  (에탄올이 없으면 고갈될 것도 없다)  <- 모델 B 문제 해결
    ★ phi->1 에서 0  (순수 에탄올은 조성 구배 불가)
    ★ ~40% 부근 최대
    이 형태는 유도된 것이며 실험에 맞춰 조정하지 않았다.
    """
    if eps is None:
        eps = EPS_DEPLETION
    phi = np.clip(phi, 0.0, 1.0)
    excess = (ALPHA - 1.0) * phi * (1.0 - phi) / (1.0 + (ALPHA - 1.0) * phi)
    return np.minimum(eps * excess, phi)


def marangoni_strength(phi, eps=None):
    """무차원 마랑고니 세기 = |gamma(phi - dphi_edge) - gamma(phi)| / gamma(0).

    모델 A와의 차이: A는 dphi_edge = phi (완전 고갈)로 두어 단조증가했다.
    모델 B와의 차이: B는 dphi_edge ∝ phi 로 두어 phi->0에서 |dgamma/dphi|가
        폭발, "에탄올 5%만 넣어도 링 소멸"이라는 반증된 예측을 냈다.
    """
    phi = np.clip(phi, 0.0, 1.0)
    dphi = preferential_depletion(phi, eps)
    dg = surface_tension(np.clip(phi - dphi, 0.0, 1.0)) - surface_tension(phi)
    return abs(float(dg)) / GAMMA_WATER


def raoult_step(V_e, V_w, dtau):
    """2성분 증발 한 스텝. 부피는 무차원(초기 총합=1).

        dV_e/dtau = -alpha*phi ,  dV_w/dtau = -(1-phi)
    정규화: 순수 물(phi=0)이 tau=1에 완전 건조.
    ★ 결과적으로 phi(t)가 저절로 감소하며, 저농도일수록 건조 후반부
      (=침착이 결정되는 시점)에 에탄올이 이미 사라져 있다.
    """
    V = V_e + V_w
    if V <= 1e-9:
        return 0.0, 0.0, 0.0
    phi = V_e / V
    V_e = max(V_e - dtau * ALPHA * phi, 0.0)
    V_w = max(V_w - dtau * (1.0 - phi), 0.0)
    V2 = V_e + V_w
    return V_e, V_w, (V_e / V2 if V2 > 1e-9 else 0.0)


def lambda_exponent(theta_deg):
    theta = np.radians(theta_deg)
    return (np.pi - 2 * theta) / (2 * (np.pi - theta))


def droplet_height(r, R, theta_deg):
    theta = np.radians(theta_deg)
    h0 = R * np.tan(theta / 2)
    rn = np.clip(r / R, 0.0, 1.0)
    return h0 * (1 - rn ** 2)


def evaporation_flux(r, R, theta_deg, phi, eps=1e-3):
    lam = lambda_exponent(theta_deg)
    rn = np.clip(r / R, 0.0, 1.0 - eps)
    return evaporation_rate(phi) * (1 - rn ** 2) ** (-lam)


def pinning_force(phi, dcos=0.15):
    """접촉선 고정력 (상대) F_pin ∝ gamma(phi)*(cos_r - cos_a).
    gamma 72 -> 25 mN/m 이면 고정력도 약 1/3."""
    return surface_tension(phi) * dcos / GAMMA_WATER


if __name__ == "__main__":
    print("=" * 68)
    print("physics.py v2 자체 검증")
    print("=" * 68)

    print("\n[1] 가장자리 고갈 dphi_edge(phi) — phi->0 에서 0이 되는가?")
    print(f"{'phi':>6} {'gamma(mN/m)':>12} {'dphi_edge':>11} {'Marangoni':>11}")
    for p in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        print(f"{p:>6.2f} {surface_tension(p)*1000:>12.1f} "
              f"{float(preferential_depletion(p)):>11.4f} "
              f"{marangoni_strength(p):>11.4f}")

    print("\n[2] 세 모델의 마랑고니 세기 형태 비교")
    print(f"{'phi':>6} {'A (v1)':>10} {'B (순진)':>10} {'C (v2)':>10}")
    for p in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        A = abs(surface_tension(0) - surface_tension(p)) / GAMMA_WATER * p ** 1.5
        B = abs(dgamma_dphi(p)) * p / GAMMA_WATER
        C = marangoni_strength(p)
        print(f"{p:>6.2f} {A:>10.4f} {B:>10.4f} {C:>10.4f}")

    print("\n[3] 건조 중 조성 phi(t) — 라울 적분")
    print(f"{'phi0':>6} {'50% dry':>9} {'75% dry':>9} {'90% dry':>9} {'95% dry':>9}")
    for p0 in [0.05, 0.10, 0.30, 0.50, 0.70]:
        Ve, Vw = p0, 1 - p0
        dt, out = 1e-4, []
        for m in [0.5, 0.25, 0.10, 0.05]:
            while Ve + Vw > m and Ve + Vw > 1e-6:
                Ve, Vw, _ = raoult_step(Ve, Vw, dt)
            out.append(Ve / max(Ve + Vw, 1e-12))
        print(f"{p0:>6.2f} {out[0]*100:>8.1f}% {out[1]*100:>8.1f}% "
              f"{out[2]*100:>8.1f}% {out[3]*100:>8.1f}%")

    print("\n[4] 접촉선 고정력 (상대)")
    for p in [0.0, 0.30, 0.50, 0.70]:
        print(f"  phi={p:.2f}: F_pin = {pinning_force(p)/pinning_force(0)*100:>5.1f}% of water")
    print("\n검증 완료.")
