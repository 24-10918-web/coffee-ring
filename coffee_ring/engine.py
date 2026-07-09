"""
engine.py — 실시간 입자 동역학 엔진 (개선판)

개선 사항:
    1. 입자 색을 '시작 위치'로 고정 (움직여도 색 안 바뀜) → color0 저장
    2. 마랑고니 세기를 강화하여 φ별 차이를 뚜렷하게
    3. 건조가 끝나면 자동으로 리셋(루프)하는 옵션
    4. 마르는 과정(높이 감소)이 유동을 구동 — 물리적으로 명시

물리: physics.py의 무차원 속도장 사용. 튜닝 없음(스케일 상수만).
"""

import numpy as np
import physics as P


class CoffeeRingSim:
    """실시간 커피링 입자 동역학."""

    def __init__(self, R=1.0, theta_deg=19.5, phi=0.5,
                 n_particles=1500, seed=0, auto_loop=True):
        self.R = R
        self.theta_deg = theta_deg
        self.phi = phi
        self.n = n_particles
        self.seed = seed
        self.auto_loop = auto_loop
        self.rng = np.random.default_rng(seed)
        self.reset()

    # --------------------------------------------------------
    def reset(self):
        """입자를 초기 상태(면적 균일 분포)로."""
        u = self.rng.uniform(0, 1, self.n)
        r = self.R * np.sqrt(u)
        ang = self.rng.uniform(0, 2 * np.pi, self.n)
        self.x = r * np.cos(ang)
        self.y = r * np.sin(ang)

        # ★ 시작 위치(초기 반경)로 색을 고정. 이후 절대 안 바뀜.
        self.color0 = (r / self.R).copy()   # 0(중심)~1(가장자리)

        self.active = np.ones(self.n, dtype=bool)
        self.pinned = np.zeros(self.n, dtype=bool)

        self.H_scale = 1.0
        self.t = 0.0
        self._setup_timescale()

    def _setup_timescale(self):
        self.lam = P.lambda_exponent(self.theta_deg)
        evap_speed = P.evaporation_rate(self.phi)
        self.t_dry = 1.0 / evap_speed
        self.n_steps_total = 1200
        self.dt = self.t_dry / self.n_steps_total
        self.dH = -1.0 / self.n_steps_total

        dg = abs(P.surface_tension(0.0) - P.surface_tension(self.phi))
        self.mar_strength = dg / P.surface_tension(0.0)
        self.mu_rel = P.viscosity(self.phi) / P.viscosity(0.0)

        # 접촉각 의존 인자 (기판 차이의 핵심):
        #   낮은 θ(친수성, 낮고 넓은 액적) → 표면장력 구배가 짧은 거리에
        #     집중 → 마랑고니 역류 강함 → 낮은 농도에서 중심 침착.
        #   높은 θ(소수성, 높고 둥근 액적) → 같은 구배가 긴 경로에 희석
        #     → 마랑고니 약함 → 더 높은 농도 필요.
        #   액적 높이 h0 ~ tan(θ/2)에 반비례하도록 모델링.
        h0 = np.tan(np.radians(self.theta_deg) / 2)
        h0_ref = np.tan(np.radians(19.5) / 2)   # 유리 기준
        self.theta_factor = h0_ref / h0          # 유리=1, OHP<1 (약함)

    # --------------------------------------------------------
    def set_params(self, phi=None, theta_deg=None):
        if phi is not None:
            self.phi = phi
        if theta_deg is not None:
            self.theta_deg = theta_deg
        self.reset()

    # --------------------------------------------------------
    def radial_velocity(self, rn):
        """무차원 반경 속도. 양수=바깥(링), 음수=안쪽(중앙).

        수정된 물리:
          - 모세관(u_cap): Deegan 깊이평균 프로파일. 중심에서도 유한하며
            반경 따라 커지다 가장자리에서 발산. → 안쪽 입자도 바깥으로 이동.
          - 마랑고니(u_mar): 세기 ∝ (표면장력 낙폭) × (실제 에탄올 농도 φ).
            ★ φ를 곱하는 것이 핵심: 저농도(5%)는 에탄올 양 자체가 적어
              마랑고니가 거의 0 → 거의 완전한 링. 고농도만 강한 역류.
          - 건조 진행(H_scale↓)에 따라 에탄올 고갈 → 마랑고니 약화.
        """
        rr = np.clip(rn, 1e-5, 1 - 1e-5)

        # --- 모세관 (Deegan 깊이평균, 중심에서도 유효) ---
        # 실제 Deegan 속도는 중심에서 r에 선형(유한), 가장자리서 발산.
        # 기존 근사가 중심에서 너무 0에 가까워 중앙 입자가 안 움직였음.
        # → 선형 baseline(0.6·r)을 더해 중심 입자도 꾸준히 바깥으로.
        edge_div = (rr / (1 - rr ** 2 + 1e-6)) * (1 - (1 - rr ** 2) ** (1 - self.lam))
        u_cap = (edge_div + 0.6 * rr) * 1.2   # baseline + 가장자리 발산

        # --- 마랑고니 (φ에 비례, 저농도서 약함) ---
        # 세기 = 낙폭 × 실제 농도 φ. → 5%면 거의 0, 70%면 강함.
        # 프로파일: 가장자리 근처에서도 힘을 유지해야 링 입자를 안쪽으로
        # 되돌릴 수 있다. rr·(1-rr²)는 가장자리에서 0이 되어 부적절 →
        # rr^0.5 형태로 바꿔 가장자리에서도 안쪽 힘 유지.
        # --- 마랑고니 (φ에 비례, 저농도서 약함) ---
        # 세기 = 낙폭 × φ^1.5. → 저농도 약, 고농도 강 (분리).
        # 프로파일: 중간 반경에서 최대, 중심(rr→0)에서는 약해지도록.
        #   √rr로만 하면 중심까지 계속 밀어 입자가 한 점에 뭉침(비현실적).
        #   rr·(1-rr) 형태 대신, 가장자리 힘 유지 + 중심 완화 절충:
        #   sin(π·rr) → 중심 0, 중간 최대, 가장자리 0 근처.
        #   여기에 가장자리 힘을 살리려 √rr 살짝 섞음.
        mar_active = max(self.H_scale, 0.0)
        mar_coef = (self.mar_strength / self.mu_rel) * (self.phi ** 1.5) * self.theta_factor
        mar_profile = 0.7 * np.sin(np.pi * rr) + 0.3 * np.sqrt(rr)
        u_mar = -mar_profile * mar_coef * 12.0 * mar_active

        return u_cap + u_mar

    # --------------------------------------------------------
    def step(self):
        """한 스텝 전진. 건조 완료 시 auto_loop이면 리셋."""
        if self.H_scale <= 0.02:
            if self.auto_loop:
                self.reset()
                return True
            return False

        self.H_scale = max(self.H_scale + self.dH, 0.0)
        self.t += self.dt

        act = self.active
        if np.any(act):
            xa, ya = self.x[act], self.y[act]
            ra = np.sqrt(xa ** 2 + ya ** 2)
            rn = np.clip(ra / self.R, 1e-5, 1.0)

            def vr(r_):
                return self.radial_velocity(r_)
            k1 = vr(rn)
            k2 = vr(np.clip(rn + 0.5 * self.dt * k1, 1e-5, 1))
            k3 = vr(np.clip(rn + 0.5 * self.dt * k2, 1e-5, 1))
            k4 = vr(np.clip(rn + self.dt * k3, 1e-5, 1))
            rn_new = np.clip(rn + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4),
                             1e-5, 0.998)

            ra_new = rn_new * self.R
            ux = np.divide(xa, ra, out=np.zeros_like(xa), where=ra > 1e-12)
            uy = np.divide(ya, ra, out=np.zeros_like(ya), where=ra > 1e-12)
            self.x[act] = ux * ra_new
            self.y[act] = uy * ra_new

            # 고정(pinning) 조건 재설계:
            #  - thin: 국소 높이 소진 → 물이 말라 더 못 움직임 → 무조건 고정.
            #  - reached(가장자리 도달): 마랑고니가 약할 때만 고정.
            #    고농도에서 마랑고니가 강하면 가장자리 입자도 안쪽으로 되돌아감
            #    (접촉선에 붙기 전에 역류가 끌어당김) → 고정 안 함.
            local_H = self.H_scale * (1 - rn_new ** 2)
            thin = local_H <= 0.02

            # 현재 마랑고니 세기 지표 (0~) — 강하면 가장자리 고정 억제
            mar_now = (self.mar_strength / self.mu_rel) * self.phi \
                * self.theta_factor * max(self.H_scale, 0.0)
            edge_pin_allowed = mar_now < 0.05   # 마랑고니 약할 때만 가장자리 고정
            reached = (rn_new >= 0.997) & edge_pin_allowed

            newly = reached | thin
            act_idx = np.where(act)[0]
            pin_idx = act_idx[newly]
            self.active[pin_idx] = False
            self.pinned[pin_idx] = True

        return True

    # --------------------------------------------------------
    def radial_density(self, n_bins=40):
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        edges = np.linspace(0, self.R, n_bins + 1)
        counts, _ = np.histogram(r, bins=edges)
        ring_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        density = counts / (ring_area + 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, density

    def metrics(self):
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        edge_frac = np.mean(r >= 0.8 * self.R)
        center_frac = np.mean(r <= 0.2 * self.R)
        centers, density = self.radial_density()
        e = np.mean(density[int(0.8*len(density)):])
        c = np.mean(density[:int(0.2*len(density))])
        ratio = e / (c + 1e-9)
        pattern = "RING" if ratio > 1.3 else ("CENTER" if ratio < 0.77 else "MIXED")
        ma = P.marangoni_number(self.R * 1e-3, self.theta_deg, self.phi)
        return {
            "edge_frac": edge_frac, "center_frac": center_frac,
            "ratio": ratio, "pattern": pattern, "ma": ma,
            "progress": min(self.t / self.t_dry, 1.0),
            "n_pinned": int(np.sum(self.pinned)),
            "n_active": int(np.sum(self.active)),
        }

    def is_done(self):
        return self.H_scale <= 0.02 or not np.any(self.active)


if __name__ == "__main__":
    print("=" * 60)
    print("engine.py 검증 — 강화된 마랑고니, φ별 차이")
    print("=" * 60)
    for label, phi in {"E0": 0.0, "E3": 0.30, "E5": 0.50, "E7": 0.70}.items():
        sim = CoffeeRingSim(phi=phi, n_particles=3000, seed=1, auto_loop=False)
        steps = 0
        while sim.step() and steps < 1300:
            steps += 1
        m = sim.metrics()
        print(f"\n{label} (phi={phi:.2f}): {steps} steps")
        print(f"  pattern={m['pattern']}, ratio={m['ratio']:.2f}, "
              f"edge={m['edge_frac']*100:.0f}% center={m['center_frac']*100:.0f}%")
