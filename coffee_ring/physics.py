"""
physics.py — 커피링 · 중심 침착 시뮬레이션의 물리 함수 모듈

설계 원칙:
    - 모든 파라미터는 이론값·문헌값. 실험 맞춤 튜닝 없음.
    - 축대칭 (모든 물리량이 반경 r에만 의존).
    - 표면장력 γ(φ)는 비선형 (에탄올-물 혼합 문헌 데이터 보간).
    - 고농도(최대 70%)까지 다루므로 μ(φ), J(φ)를 φ의 함수로 반영.

농도 표기: φ는 에탄올 부피분율 (0.0 ~ 0.7).
    실험 라벨과의 대응: E0=0.0, E3=0.30, E5=0.50, E7=0.70.

참고 문헌 개념:
    - Deegan et al. (1997): 접촉선 고정 하 증발 플럭스 발산, 커피링 형성.
    - Hu & Larson (2006): 마랑고니 역류에 의한 커피링 억제.
    - Vazquez et al. (1995): 에탄올-물 혼합 표면장력 데이터.
"""

import numpy as np
from scipy.interpolate import interp1d


# ============================================================
# 물리 상수 (전부 문헌값, 상온 ~25°C 기준)
# ============================================================
GAMMA_WATER = 0.072      # 물 표면장력 [N/m]
GAMMA_ETHANOL = 0.022    # 에탄올 표면장력 [N/m]
MU_WATER = 1.00e-3       # 물 점성 [Pa·s]
MU_ETHANOL = 1.20e-3     # 에탄올 점성 [Pa·s] (물보다 약간 높음)
RHO = 1.0e3              # 밀도 [kg/m^3] (물 근사)
D_PARTICLE = 1.0e-12     # 커피 입자 확산계수 [m^2/s] (입자 크기 기반)


# ============================================================
# 1. 표면장력 γ(φ) — 비선형 (문헌 데이터 보간)
# ============================================================
# 에탄올-물 혼합 표면장력. 부피분율 φ에 대한 실측 경향.
# 저농도에서 급락 → 고농도로 갈수록 완만해지는 비선형 형태.
# (Vazquez et al. 1995 등의 경향을 반영한 대표값. 실측 데이터 보간.)
_PHI_DATA = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40,
                      0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
_GAMMA_DATA = np.array([0.0720, 0.0560, 0.0480, 0.0400, 0.0350, 0.0310,
                        0.0285, 0.0265, 0.0250, 0.0238, 0.0228, 0.0220])

# 단조 3차 보간 (PCHIP 대신 안전하게 cubic + 범위 클램프)
_gamma_interp = interp1d(_PHI_DATA, _GAMMA_DATA, kind='cubic',
                         bounds_error=False,
                         fill_value=(_GAMMA_DATA[0], _GAMMA_DATA[-1]))


def surface_tension(phi):
    """표면장력 γ(φ) [N/m]. 비선형 보간.

    phi: 에탄올 부피분율 (스칼라 또는 배열, 0~1)
    """
    phi = np.clip(phi, 0.0, 1.0)
    return float(_gamma_interp(phi)) if np.isscalar(phi) else _gamma_interp(phi)


def dgamma_dphi(phi, h=1e-4):
    """dγ/dφ — 표면장력의 농도 미분 (수치 미분).

    마랑고니 역류의 구동력 계산에 사용.
    저농도에서 크게 음수(급락), 고농도에서 완만.
    """
    phi = np.clip(phi, h, 1.0 - h)
    return (surface_tension(phi + h) - surface_tension(phi - h)) / (2 * h)


# ============================================================
# 2. 점성 μ(φ) — 에탄올-물 혼합 (비선형)
# ============================================================
# 에탄올-물 혼합 점성은 순수 성분보다 높은 봉우리를 가짐(수소결합).
# 중간 농도(~40%)에서 최대. 단순화한 2차 형태로 근사.
def viscosity(phi):
    """혼합 점성 μ(φ) [Pa·s].

    양 끝(순수 물/에탄올)은 각 성분값, 중간은 봉우리.
    peak_factor는 40% 부근 최대 점성을 반영한 문헌 경향값.
    """
    phi = np.clip(phi, 0.0, 1.0)
    mu_linear = MU_WATER * (1 - phi) + MU_ETHANOL * phi
    # 중간 농도 봉우리 (혼합 점성 증가). 최대 ~2.4배 at phi=0.4 (문헌 경향).
    peak = 1.0 + 1.4 * phi * (1 - phi) * 4.0 * np.exp(-((phi - 0.4) ** 2) / 0.08)
    return mu_linear * peak


# ============================================================
# 3. 증발률 J₀(φ) — 에탄올이 물보다 빠르게 증발
# ============================================================
# 에탄올 증기압이 물의 약 2.5배 → φ 클수록 초기 증발 가속.
# 상대 증발률만 사용 (J0_water를 1로 정규화).
J0_WATER_REL = 1.0
EVAP_RATIO = 2.5   # 에탄올/물 증발률 비 (증기압 비 기반)


def evaporation_rate(phi):
    """상대 증발률 J₀(φ) (물=1로 정규화).

    혼합물 초기 증발률. φ 클수록 큼(에탄올 기여).
    선형 혼합 근사: J0 = (1-φ)·1 + φ·EVAP_RATIO
    """
    phi = np.clip(phi, 0.0, 1.0)
    return J0_WATER_REL * (1 - phi) + EVAP_RATIO * phi


# ============================================================
# 4. 접촉각 → λ (증발 플럭스 지수)
# ============================================================
def lambda_exponent(theta_deg):
    """증발 플럭스 지수 λ = (π - 2θ)/(2(π - θ)).

    Deegan(1997) 접촉선 고정 조건.
    theta_deg: 접촉각 [도]. θ < 90° 범위에서 유효.
    """
    theta = np.radians(theta_deg)
    return (np.pi - 2 * theta) / (2 * (np.pi - theta))


# ============================================================
# 5. 액적 형상 h(r) — 구면 캡
# ============================================================
def droplet_height(r, R, theta_deg):
    """액적 높이 프로파일 h(r) [상대 단위].

    구면 캡 근사: h(r) = h0 · [1 - (r/R)^2]
    h0 ≈ R·tan(θ/2) (작은 θ에서).
    r: 반경 좌표 (배열), R: 액적 반경, theta_deg: 접촉각.
    """
    theta = np.radians(theta_deg)
    h0 = R * np.tan(theta / 2)
    rn = np.clip(r / R, 0.0, 1.0)
    return h0 * (1 - rn ** 2)


# ============================================================
# 6. 증발 플럭스 J(r) — 가장자리 발산
# ============================================================
def evaporation_flux(r, R, theta_deg, phi, eps=1e-3):
    """국소 증발 플럭스 J(r).

    J(r) = J0(φ) · [1 - (r/R)^2]^(-λ)
    가장자리(r→R)에서 발산. eps로 수치적 특이점 방지.
    """
    lam = lambda_exponent(theta_deg)
    J0 = evaporation_rate(phi)
    rn = np.clip(r / R, 0.0, 1.0 - eps)
    return J0 * (1 - rn ** 2) ** (-lam)


# ============================================================
# 7. 모세관 흐름 u_cap(r) — 바깥 방향 (링 생성)
# ============================================================
def capillary_velocity(r, R, theta_deg, phi, eps=1e-3):
    """깊이 평균 모세관 반경 속도 u_cap(r).

    질량 보존(연속방정식)에서:
        u_cap(r) = (1/(r·h)) · ∫₀ʳ [J(r')/ρ] · r' dr'
    가장자리로 향하는 흐름(양수 = 바깥 방향).
    수치 적분(사다리꼴)으로 계산.
    """
    r = np.atleast_1d(r).astype(float)
    h = droplet_height(r, R, theta_deg)
    J = evaporation_flux(r, R, theta_deg, phi, eps=eps)

    # 누적 적분 ∫₀ʳ J(r')·r' dr'
    integrand = J * r / RHO
    cumint = np.zeros_like(r)
    cumint[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))

    # u_cap = cumint / (r·h), 특이점 방지
    denom = r * h
    u = np.divide(cumint, denom, out=np.zeros_like(cumint),
                  where=denom > 1e-12)
    return u


# ============================================================
# 8. 마랑고니 역류 u_mar(r) — 안쪽 방향 (링 억제)
# ============================================================
def marangoni_velocity(r, R, theta_deg, phi, eps=1e-3):
    """깊이 평균 마랑고니 반경 속도 u_mar(r) (표면층 기여).

    u_mar ∝ (dγ/dr) · (h / μ),  dγ/dr = (dγ/dφ)·(dφ/dr)
    에탄올이 가장자리에서 먼저 증발 → 가장자리 φ 낮음 → γ 높음
    → 표면이 중심으로 당겨짐 (음수 = 안쪽 방향).

    dφ/dr 모델: 증발이 가장자리 집중이므로 가장자리에서 φ가 더 빨리 감소.
    간단화 — dφ/dr ∝ 증발 플럭스 분포에 비례한다고 가정하고,
    부호는 가장자리로 갈수록 φ 감소(음의 구배는 중심방향 힘).
    """
    r = np.atleast_1d(r).astype(float)
    h = droplet_height(r, R, theta_deg)
    mu = viscosity(phi)

    # 가장자리 집중 증발 → 국소 에탄올 고갈 구배의 대리 지표.
    # 정규화된 증발 플럭스 형태를 dφ/dr의 크기로 사용.
    lam = lambda_exponent(theta_deg)
    rn = np.clip(r / R, 0.0, 1.0 - eps)
    # 반경에 따라 증가하는 프로파일 (가장자리에서 구배 최대)
    grad_shape = rn * (1 - rn ** 2) ** (-lam)

    dg_dphi = dgamma_dphi(phi)      # 음수 (φ 증가 시 γ 감소)
    # dφ/dr: 가장자리로 갈수록 에탄올 고갈(φ 감소) → dφ/dr < 0
    # 세기는 φ가 클수록(넣은 에탄올 많을수록) 큼
    dphi_dr = -grad_shape * phi / R

    # dγ/dr = dγ/dφ · dφ/dr  →  (음)·(음) = 양? 부호 주의:
    #   가장자리 γ 높음 → 중심방향 힘(음). 아래서 -로 맞춤.
    dgamma_dr = dg_dphi * dphi_dr    # 부호 포함
    # 표면 응력 → 깊이평균 속도: u ~ (dγ/dr)·h/(2μ), 중심방향이 음수가 되도록 -부호
    u = -(dgamma_dr) * h / (2 * mu)
    return u


# ============================================================
# 9. 알짜 속도 u_net(r) = u_cap + u_mar
# ============================================================
def net_velocity(r, R, theta_deg, phi, eps=1e-3):
    """알짜 반경 속도. 양수=바깥(링), 음수=안쪽(중심 침착)."""
    u_cap = capillary_velocity(r, R, theta_deg, phi, eps=eps)
    u_mar = marangoni_velocity(r, R, theta_deg, phi, eps=eps)
    return u_cap + u_mar, u_cap, u_mar


# ============================================================
# 10. 마랑고니 수 Ma (무차원 경쟁 지표)
# ============================================================
def marangoni_number(R, theta_deg, phi):
    """Ma = |Δγ| · R_phys / (μ · D).

    Δγ: 가장자리-중심 표면장력 차 (φ 고갈에 의한).
    정성적 지표로만 사용 (엄밀한 임계점 아님).
    R은 물리 반경[m] 필요 — 여기선 상대 크기로 R_phys=R 사용.
    """
    # 가장자리에서 에탄올이 거의 고갈된다고 보면 Δγ ≈ γ(0) - γ(φ)
    dgamma = abs(surface_tension(0.0) - surface_tension(phi))
    mu = viscosity(phi)
    R_phys = R if R > 1e-4 else 1e-3   # 물리 스케일 보정 (m)
    return dgamma * R_phys / (mu * D_PARTICLE)


# ============================================================
# 자체 검증 (직접 실행 시)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("physics.py 자체 검증")
    print("=" * 60)

    # 실험 농도 라벨
    cases = {"E0": 0.0, "E3": 0.30, "E5": 0.50, "E7": 0.70}

    print("\n[1] 표면장력 γ(φ) — 비선형 확인")
    for label, phi in cases.items():
        print(f"  {label} (φ={phi:.2f}): γ = {surface_tension(phi)*1000:.1f} mN/m")
    # 저농도 급락 확인
    print(f"  급락 확인: γ(0.05)={surface_tension(0.05)*1000:.1f}, "
          f"γ(0.10)={surface_tension(0.10)*1000:.1f} mN/m")

    print("\n[2] dγ/dφ — 저농도서 크고 고농도서 완만한지")
    for phi in [0.05, 0.30, 0.50, 0.70]:
        print(f"  φ={phi:.2f}: dγ/dφ = {dgamma_dphi(phi)*1000:.1f} mN/m per unit φ")

    print("\n[3] 점성 μ(φ) — 중간 봉우리 확인")
    for phi in [0.0, 0.2, 0.4, 0.6, 0.7, 1.0]:
        print(f"  φ={phi:.2f}: μ = {viscosity(phi)*1000:.3f} mPa·s")

    print("\n[4] 증발률 J₀(φ) — φ 클수록 증가")
    for label, phi in cases.items():
        print(f"  {label} (φ={phi:.2f}): J0(rel) = {evaporation_rate(phi):.2f}")

    print("\n[5] λ(θ) — 기판별")
    for name, th in [("유리", 19.5), ("OHP", 50.0)]:
        print(f"  {name} (θ={th}°): λ = {lambda_exponent(th):.3f}")

    print("\n[6] Ma 수 — φ 클수록 증가 (경향)")
    R = 1e-3
    for label, phi in cases.items():
        print(f"  {label} (φ={phi:.2f}): Ma = {marangoni_number(R, 19.5, phi):.2e}")

    print("\n[7] 속도장 부호 확인 (유리, φ=0.5, 반경 중간/가장자리)")
    r = np.linspace(1e-4, R, 50)
    u_net, u_cap, u_mar = net_velocity(r, R, 19.5, 0.5)
    print(f"  u_cap 부호(가장자리): {'양(바깥)' if u_cap[-2] > 0 else '음'}")
    print(f"  u_mar 부호(가장자리): {'음(안쪽)' if u_mar[-2] < 0 else '양'}")
    print(f"  중간 반경 u_net: {u_net[25]:.2e}, 가장자리 u_net: {u_net[-2]:.2e}")

    print("\n검증 완료.")
