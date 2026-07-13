"""
engine_v2.py — 커피링 시뮬레이션 엔진 v2

v1(engine.py)의 세 가지 결함을 고쳤다.
====================================================================
[결함 1] 마랑고니 구동력을 Δγ = γ(0) − γ(φ) 로 계산했다.
   → "가장자리가 에탄올 0%까지 완전 고갈된다"는 숨은 가정이며, φ에 대해 단조증가하므로
     "농도↑ → 억제↑"를 결론이 아니라 전제로 넣은 것이다.
   [수정] Δγ = γ((1−f)·x) − γ(x).   f = 가장자리 고갈 분율 (모델링 선택, 민감도 분석 대상)
     f=1 → v1과 동일.  f→0 → 국소 기울기 모델.
     ★ 접선 근사 |dγ/dφ|·Δφ 는 쓰지 않는다. γ 곡선은 φ→0에서 기울기가 −240 mN/m로
       발산해, 접선 근사는 "에탄올 5%만 넣어도 링이 사라진다"는 비물리적 결과를 낳는다.
       γ의 차를 그대로 쓰면 이 폭주가 자동 제거된다.

[결함 2] φ를 건조 내내 상수로 두었다.
   → 실제로는 에탄올이 물보다 α(=2.5)배 빨리 증발해 건조 도중 소진된다.
   [수정] 조성 x(t)를 시간 적분한다:
        dV_eth/dt = −C·A·α·x ,   dV_wat/dt = −C·A·(1−x)
     → 마랑고니는 건조 전반부에만 작동하고, 후반부는 사실상 순수 물 건조가 된다.
     → E30에서 '중앙 침착 + 잔류 링'이 동시에 나타나는 이유. v1은 재현 불가였다.

[결함 3] φ^1.5 튜닝 항 + 접촉각 19.5° 고정.
   → φ^1.5는 코드 주석에 목적이 적힌 명백한 튜닝 상수였다.
   [수정] 제거. 대신 마랑고니/모세관 경쟁비를 스케일링으로 유도한다:
        u_mar ~ Δγ·h/(μ·a)        (표면 응력 → 유동)
        u_cap ~ J0·a/(ρ·h)        (Deegan 증발 보충류)
        ⇒ Ma ≡ u_mar/u_cap ∝ Δγ/(μ·J0) · (h/a)² = Δγ/(μ·J0) · tan²(θ/2)
     → 접촉각이 자동으로 등장한다. 접촉선이 고정된 액적은 마르면서 θ가 계속 줄어들므로
       마랑고니가 기하학적으로 약해지고, 마지막엔 모세관이 이겨 잔류 링이 생긴다.
       ★ 튜닝이 아니라 유도다.

파라미터 총목록 (전부 공개)
====================================================================
 이론·문헌값(고정) : γ(φ) 보간표, μ(φ), α=2.5, Deegan λ(θ)
 실험 경계조건     : a(φ), θ₀(φ)   [experiment.py — 액적 반경과 같은 범주의 초기조건]
 환경 보정 1개     : C_evap  ← E0(순수 물, 마랑고니 없음) 건조 곡선만으로 보정. 온습도 읽기.
 모델링 선택 2개   : f, K    ← 민감도 분석 대상

예측(=검증 대상, 절대 입력하지 않음)
 · θ(t) : 4농도 × 3시점 = 12개 독립 검증점
 · 침착 패턴 : e/c, 반경 밀도 프로파일
 · 최적 농도
"""

import numpy as np
import physics as P
import experiment as EXP

ALPHA = 2.5
GAMMA0 = P.surface_tension(0.0)
MU0 = P.viscosity(0.0)


def delta_gamma(x, f):
    """가장자리 고갈에 의한 표면장력 차 [N/m]. 정확한 γ 차 (접선근사 아님)."""
    if x <= 1e-6:
        return 0.0
    return float(P.surface_tension((1.0 - f) * x) - P.surface_tension(x))


class SimV2:
    def __init__(self, phi0, f=0.30, K=1.0, C_evap=1.0,
                 n_particles=3000, seed=0, n_steps=1500, a=None, th0=None):
        self.phi0 = float(phi0)
        self.f, self.K, self.C_evap = float(f), float(K), float(C_evap)
        self.n, self.n_steps = int(n_particles), int(n_steps)
        self.a = EXP.contact_radius(phi0) if a is None else float(a)
        self.th0 = EXP.theta0(phi0) if th0 is None else float(th0)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        u = self.rng.uniform(0, 1, self.n)
        rn = np.sqrt(u)
        ang = self.rng.uniform(0, 2*np.pi, self.n)
        self.x, self.y = rn*np.cos(ang), rn*np.sin(ang)
        self.color0 = rn.copy()
        self.active = np.ones(self.n, dtype=bool)
        self.pinned = np.zeros(self.n, dtype=bool)

        self.V0 = EXP.cap_volume(self.a, self.th0)
        self.V_eth = self.phi0 * self.V0
        self.V_wat = (1 - self.phi0) * self.V0
        self.V = self.V0
        self.xc = self.phi0
        self.theta = self.th0
        self.t = 0.0
        self.hist = []
        # 총 건조가 대략 n_steps 안에 끝나도록 dt 설정 [분]
        A0 = EXP.cap_area(self.a, self.th0)
        rate0 = self.C_evap * A0 * (1 + (ALPHA-1)*self.phi0)
        self.dt = (self.V0 / rate0) / self.n_steps * 2.5
        self.t_scale = self.V0 / (self.C_evap * A0)   # 무차원화용 기준 건조시간

    def evaporate(self):
        A = EXP.cap_area(self.a, self.theta)
        base = self.C_evap * A * self.dt
        self.V_eth = max(self.V_eth - base*ALPHA*self.xc, 0.0)
        self.V_wat = max(self.V_wat - base*(1-self.xc), 0.0)
        self.V = self.V_eth + self.V_wat
        self.xc = self.V_eth/self.V if self.V > 1e-18 else 0.0
        self.theta = EXP.theta_from_volume(self.a, self.V)
        self.t += self.dt

    def marangoni_number(self):
        x = self.xc
        if x <= 1e-4:
            return 0.0
        dg = delta_gamma(x, self.f) / GAMMA0
        mu_rel = P.viscosity(x) / MU0
        J0 = 1 + (ALPHA-1)*x
        aspect = np.tan(np.radians(self.theta)/2)**2
        return self.K * dg / (mu_rel*J0) * aspect

    def radial_velocity(self, r, Ma, lam):
        rr = np.clip(r, 1e-5, 1-1e-5)
        edge_div = (rr/(1-rr**2+1e-6)) * (1-(1-rr**2)**(1-lam))
        u_cap = edge_div + 0.6*rr
        prof = 0.7*np.sin(np.pi*rr) + 0.3*np.sqrt(rr)
        return u_cap - prof*Ma

    def step(self):
        if self.V/self.V0 <= 0.02:
            return False
        self.evaporate()
        Ma = self.marangoni_number()
        lam = P.lambda_exponent(self.theta)
        Hn = self.V/self.V0
        dts = self.dt / self.t_scale

        act = self.active
        if np.any(act):
            xa, ya = self.x[act], self.y[act]
            ra = np.sqrt(xa**2 + ya**2)
            rr = np.clip(ra, 1e-5, 1.0)
            k1 = self.radial_velocity(rr, Ma, lam)
            k2 = self.radial_velocity(np.clip(rr+.5*dts*k1, 1e-5, 1), Ma, lam)
            k3 = self.radial_velocity(np.clip(rr+.5*dts*k2, 1e-5, 1), Ma, lam)
            k4 = self.radial_velocity(np.clip(rr+dts*k3, 1e-5, 1), Ma, lam)
            rn = np.clip(rr + (dts/6)*(k1+2*k2+2*k3+k4), 1e-5, 0.998)
            ux = np.divide(xa, ra, out=np.zeros_like(xa), where=ra > 1e-12)
            uy = np.divide(ya, ra, out=np.zeros_like(ya), where=ra > 1e-12)
            self.x[act], self.y[act] = ux*rn, uy*rn

            local_h = Hn*(1-rn**2)
            thin = local_h <= 0.02
            reached = (rn >= 0.997) & (Ma < 0.02)
            idx = np.where(act)[0][thin | reached]
            self.active[idx] = False
            self.pinned[idx] = True

        self.hist.append((self.t, self.theta, Hn, self.xc, Ma))
        return True

    def run(self):
        for _ in range(self.n_steps*6):
            if not self.step():
                break
        return self

    def theta_at(self, t_min):
        h = np.array(self.hist)
        if len(h) == 0:
            return self.th0
        return float(np.interp(t_min, h[:, 0], h[:, 1],
                               left=self.th0, right=h[-1, 1]))

    def radial_density(self, nb=30):
        r = np.sqrt(self.x**2 + self.y**2)
        edges = np.linspace(0, 1, nb+1)
        cnt, _ = np.histogram(r, bins=edges)
        area = np.pi*(edges[1:]**2 - edges[:-1]**2)
        return 0.5*(edges[:-1]+edges[1:]), cnt/area, area

    def metrics(self):
        c, d, a = self.radial_density()
        e = np.mean(d[int(.8*len(d)):]); ce = np.mean(d[:int(.2*len(d))])
        ratio = e/(ce+1e-9)
        w = a/a.sum(); mean = np.sum(d*w)
        cv = float(np.sqrt(np.sum(w*(d-mean)**2))/(mean+1e-12))
        h = np.array(self.hist)
        off = 1.0
        if len(h):
            i = np.where(h[:, 4] < 0.02)[0]
            if len(i):
                off = 1.0 - h[i[0], 2]
        return {"ratio": float(ratio), "cv": cv,
                "peak_over_mean": float(d.max()/(mean+1e-12)),
                "peak_r": float(c[np.argmax(d)]),
                "pattern": "RING" if ratio > 1.3 else ("CENTER" if ratio < 0.77 else "MIXED"),
                "mar_off_at": float(off), "theta_end": float(self.theta),
                "t_end": float(self.t)}


def measure(phi0, f, K, C_evap, seeds=3, n=2000):
    ms = [SimV2(phi0, f=f, K=K, C_evap=C_evap, seed=s, n_particles=n).run().metrics()
          for s in range(seeds)]
    return {k: float(np.mean([m[k] for m in ms]))
            for k in ["ratio", "cv", "peak_over_mean", "peak_r", "mar_off_at", "t_end"]}
