"""
run_v2.py — v2 전체 실행 스크립트

  1) C_evap 보정  (E0 순수 물 건조 곡선 하나만 사용)
  2) 건조 곡선 theta(t) 예측 vs 실측  [독립 검증]
  3) 농도 스윕 → 침착 패턴, 최적 농도
  4) (f, K) 민감도 분석
  5) 차트 저장
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from engine_v2 import SimV2, measure
import experiment as EXP

# ---------- 1) C_evap 보정 ----------
def _err(C):
    return SimV2(0.0, K=0.0, C_evap=C, n_particles=30, n_steps=800).run().theta_at(15.0) \
        - EXP.THETA_T[0.00][3]
C_EVAP = brentq(_err, 1e-9, 1e-3, xtol=1e-12)

F, K = 0.35, 500.0          # 모델링 선택 (민감도 분석 대상)
BANDS = [(0.25, 300), (0.35, 500), (0.50, 700)]

if __name__ == "__main__":
    print(f"[1] C_evap = {C_EVAP:.4e} m/min  (E0 15분 접촉각만으로 보정)\n")

    print("[2] 건조 곡선 theta(t) — 예측 vs 실측")
    print(f"{'phi':>5} | {'10분 예측/실측':>20} | {'15분 예측/실측':>20}")
    for p in EXP.PHI_LIST:
        s = SimV2(p, f=F, K=0.0, C_evap=C_EVAP, n_particles=30, n_steps=900).run()
        print(f"{p*100:>4.0f}% | {s.theta_at(10):>8.2f}° / {EXP.THETA_T[p][2]:>6.2f}° "
              f"| {s.theta_at(15):>8.2f}° / {EXP.THETA_T[p][3]:>6.2f}°")

    print("\n[3] 농도 스윕")
    grid = np.arange(0, 0.72, 0.05)
    res = []
    for p in grid:
        m = measure(p, F, K, C_EVAP, seeds=4, n=2500)
        res.append([p, m['ratio'], m['cv'], m['peak_over_mean']])
        pat = "RING" if m['ratio'] > 1.3 else ("CENTER" if m['ratio'] < 0.77 else "MIXED")
        print(f"  {p*100:>4.0f}%  e/c={m['ratio']:>6.2f}  CV={m['cv']:>5.2f}  {pat}")
    res = np.array(res)

    print("\n[4] (f,K) 민감도")
    for f, k in BANDS:
        rr = np.array([[p] + [measure(p, f, k, C_EVAP, seeds=2, n=1500)[q]
                              for q in ['ratio', 'cv']] for p in grid])
        print(f"  f={f:.2f} K={k:>4.0f} -> e/c=1 at {rr[np.argmin(abs(rr[:,1]-1)),0]*100:>3.0f}%")

    # ---------- 5) 차트 ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(res[:, 0]*100, res[:, 1], 'o-', color='#2a78d6')
    ax[0].axhline(1, ls=':', c='gray'); ax[0].set_yscale('log')
    ax[0].axvspan(0, 25, alpha=.12, color='red')
    ax[0].text(12, 0.05, 'model invalid\n(low-phi blowup)', ha='center', fontsize=8, color='#a32d2d')
    ax[0].set_xlabel('ethanol (%)'); ax[0].set_ylabel('edge/center (log)')
    ax[0].set_title('v2: deposit pattern vs concentration'); ax[0].grid(alpha=.3)

    for p in EXP.PHI_LIST:
        s = SimV2(p, f=F, K=0.0, C_evap=C_EVAP, n_particles=30, n_steps=900).run()
        h = np.array(s.hist)
        ax[1].plot(h[:, 0], h[:, 1], '-', lw=1.5, label=f'E{int(p*100)} predicted')
        ax[1].plot(EXP.T_MIN, EXP.THETA_T[p], 'o--', ms=5, alpha=.6)
    ax[1].set_xlim(0, 16); ax[1].set_xlabel('time (min)'); ax[1].set_ylabel('contact angle (deg)')
    ax[1].set_title('theta(t): line = predicted, dots = measured')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig('v2_results.png', dpi=130, bbox_inches='tight')
    print("\nsaved: v2_results.png")
