"""
experiment.py — 실험 경계조건 (v2)

★ 시뮬레이션의 '입력'과 '예측'을 엄격히 구분한다.

  [입력 = 경계조건]  액적의 기하학. 물리 법칙이 아니라 초기 조건이다.
      - a(phi)      : 접촉 반경 [건조 후 얼룩 지름 / 2]  (접촉선 고정 → 건조 내내 일정)
      - theta0(phi) : 초기 접촉각 (t=0 측면 사진)
      이 둘이 초기 부피 V0를 결정한다. 실험에서 읽어올 수밖에 없는 값.
      (액적 반경 R을 실험에서 받는 것과 완전히 동일한 범주다.)

  [예측 = 검증 대상]  시뮬은 이 값들을 절대 입력받지 않는다.
      - theta(t)    : 5·10·15분 접촉각  → 4농도 x 3시점 = 12개 독립 검증점
      - 침착 패턴   : e/c 비, 반경 밀도 프로파일, 링 폭
      - 최적 농도

  [보정 상수 1개]  C_evap (실험실 온습도·기류를 반영하는 주변 환경 상수)
      → E0(순수 물, 마랑고니 없음) 건조 곡선 하나로만 보정한다.
      → E30/E50/E70의 건조 곡선은 전부 '예측'이 된다.
"""

import numpy as np

# ── 실측: 건조 후 얼룩 지름 [mm] ─────────────────────────
DIAMETER_MM = {0.00: 7.687, 0.30: 7.480, 0.50: 8.795, 0.70: 9.639}

# ── 실측: 커피링 폭 [mm] ────────────────────────────────
RING_WIDTH_MM = {0.00: 0.361, 0.30: 0.343, 0.50: 0.257, 0.70: 0.157}

# ── 실측: 접촉각 시계열 [도], 측면 사진 ──────────────────
#        0분(초기) / 5분 / 10분 / 15분
THETA_T = {
    0.00: [41.782, 23.582, 21.010, 19.094],
    0.30: [37.940, 20.247, 17.316, 15.574],
    0.50: [31.151, 18.817, 11.054, 10.154],
    0.70: [26.348, 17.438, 10.724,  6.864],
}
T_MIN = np.array([0.0, 5.0, 10.0, 15.0])

PHI_LIST = [0.00, 0.30, 0.50, 0.70]


def contact_radius(phi):
    """접촉 반경 a [m]. 실측 지름/2. (접촉선 고정 가정)"""
    d = np.interp(phi, PHI_LIST, [DIAMETER_MM[p] for p in PHI_LIST])
    return d / 2 * 1e-3


def theta0(phi):
    """초기 접촉각 [도]. 실측."""
    return float(np.interp(phi, PHI_LIST, [THETA_T[p][0] for p in PHI_LIST]))


def cap_volume(a, theta_deg):
    """구면 캡 부피 V(a, theta) [m^3].  h = a*tan(theta/2)"""
    h = a * np.tan(np.radians(theta_deg) / 2)
    return np.pi / 6 * h * (3 * a ** 2 + h ** 2)


def theta_from_volume(a, V):
    """부피 V와 접촉 반경 a로부터 접촉각 [도]을 역산 (접촉선 고정)."""
    V = max(float(V), 0.0)
    lo, hi = 1e-4, 89.9
    for _ in range(60):                    # 이분법
        mid = 0.5 * (lo + hi)
        if cap_volume(a, mid) < V:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cap_area(a, theta_deg):
    """액체-공기 계면 넓이 [m^2] (구면 캡 곡면)."""
    th = np.radians(theta_deg)
    if th < 1e-6:
        return np.pi * a ** 2
    Rs = a / np.sin(th)                    # 곡률 반경
    h = Rs * (1 - np.cos(th))
    return 2 * np.pi * Rs * h


def measured_volume_series(phi):
    """실측 theta(t) → 부피 시계열 V(t)/V0. (접촉선 고정 → theta가 부피를 읽어준다)"""
    a = contact_radius(phi)
    Vs = np.array([cap_volume(a, th) for th in THETA_T[phi]])
    return Vs / Vs[0], Vs


if __name__ == "__main__":
    print("=" * 68)
    print(" 실험 경계조건 & 유도된 부피 곡선")
    print("=" * 68)
    print(f"{'phi':>5} {'a(mm)':>7} {'th0':>7} {'V0(uL)':>8} |  V/V0  @0/5/10/15분")
    for p in PHI_LIST:
        a = contact_radius(p)
        vn, Vs = measured_volume_series(p)
        print(f"{p*100:>4.0f}% {a*1e3:>7.3f} {theta0(p):>6.1f}° "
              f"{Vs[0]*1e9:>8.1f} |  " + "  ".join(f"{v:.3f}" for v in vn))

    print("\n※ V0가 4농도에서 비슷하면(같은 피펫) → 구면캡+접촉선고정 가정이 타당함을 뒷받침")

    print("\n" + "=" * 68)
    print(" 단위 면적당 증발 플럭스 [um/min] — 이론 J0(phi)=1+1.5phi 와 비교")
    print("=" * 68)
    print(f"{'phi':>5} {'0-5분':>8} {'5-10분':>8} {'10-15분':>9} | {'이론 J0':>8} {'실측비(0-5)':>11}")
    base = None
    for p in PHI_LIST:
        a = contact_radius(p)
        Vs = np.array([cap_volume(a, th) for th in THETA_T[p]])
        fl = []
        for i in range(3):
            th_mid = 0.5 * (THETA_T[p][i] + THETA_T[p][i + 1])
            A = cap_area(a, th_mid)
            dV = Vs[i] - Vs[i + 1]
            fl.append(dV / A / 5 * 1e6)     # m/min -> um/min
        if base is None:
            base = fl[0]
        J0 = 1 + 1.5 * p
        print(f"{p*100:>4.0f}% {fl[0]:>8.1f} {fl[1]:>8.1f} {fl[2]:>9.1f} | "
              f"{J0:>8.2f} {fl[0]/base:>11.2f}")
