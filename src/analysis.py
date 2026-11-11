"""
analysis.py — Tankönyvi jóslatok, statisztika, ítéletek
Máté + Maya, 2026-11-11

Ítéletek: TEXTBOOK | ANOMALY | INVALID
"""

import numpy as np
from scipy.special import erfc
from scipy.stats import binom_test, binomtest
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ─── Kalibráció eredménye ─────────────────────────────────────────────────────

@dataclass
class CalibResult:
    f0: float           # Hz — kalibrált rezonancia
    Q: float            # minőségi tényező
    tau: float          # s — lecsengési idő (= Q / (π·f0))
    sigma: float        # lock-in zaj szintje (I komponens std)
    A0: float           # kezdeti amplitúdó
    Q_std: float        # Q szórása az ismétlések között
    f0_drift: float     # Hz — f0 sodródása a kalibráció alatt
    valid: bool         # átment-e az elfogadási kritériumon


def check_calib(result: CalibResult, cfg) -> str:
    """VALID / INVALID a kalibráció elfogadásán."""
    bw = result.f0 / result.Q  # sávszélesség
    q_tol = result.Q_std / result.Q * 100.0
    drift_ok = result.f0_drift < cfg.calib_f0_drift_frac * bw
    q_ok     = q_tol < cfg.calib_q_tol_pct
    if q_ok and drift_ok:
        return "VALID"
    reasons = []
    if not q_ok:
        reasons.append(f"Q szórás {q_tol:.1f}% > {cfg.calib_q_tol_pct}%")
    if not drift_ok:
        reasons.append(f"f0 sodródás {result.f0_drift:.4f} Hz > BW/4 = {bw/4:.4f} Hz")
    logger.error("INVALID kalibráció: %s", "; ".join(reasons))
    return "INVALID"


# ─── 1. fázis: BER tankönyvi jóslat ──────────────────────────────────────────

def predict_BER(delay_s: float, calib: CalibResult) -> float:
    """
    Tankönyvi BER jóslat szabad lecsengésnél.
    BER(t) = ½ · erfc(A(t) / (σ·√2))
    A(t) = A₀ · exp(−t/τ)
    """
    A_t = calib.A0 * np.exp(-delay_s / calib.tau)
    snr = A_t / (calib.sigma * np.sqrt(2))
    return 0.5 * float(erfc(snr))


def binomial_ci_99(n_trials: int, BER_pred: float) -> Tuple[float, float]:
    """99%-os binomiális sáv a jósolt BER köré."""
    from scipy.stats import binom
    lo = binom.ppf(0.005, n_trials, BER_pred) / n_trials
    hi = binom.ppf(0.995, n_trials, BER_pred) / n_trials
    return float(lo), float(hi)


def holm_correct(p_values: List[float], alpha: float) -> List[bool]:
    """
    Holm–Bonferroni korrekció.
    Visszaad: list[bool] — rejected[i] True ha szignifikáns
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    for rank, (i, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        if p <= threshold:
            rejected[i] = True
        else:
            break  # Holm: ha egyszer nem szignifikáns, a többi sem
    return rejected


def judge_phase1(delays_tau: List[float], measured_BERs: List[float],
                 n_trials: int, calib: CalibResult,
                 alpha: float = 0.01) -> dict:
    """
    1. fázis ítélet.
    
    TEXTBOOK: minden mért BER a 99%-os binomiális sávon belül.
    ANOMALY:  ≥2 késleltetésen szignifikánsan jobb a jóslatnál (Holm).
    """
    n = len(delays_tau)
    assert len(measured_BERs) == n

    pred_BERs = []
    in_band   = []
    p_better  = []  # egyoldali p: mért jobb mint jósolt

    for delay_frac, ber_meas in zip(delays_tau, measured_BERs):
        delay_s = delay_frac * calib.tau
        ber_pred = predict_BER(delay_s, calib)
        lo, hi   = binomial_ci_99(n_trials, ber_pred)
        in_band.append(lo <= ber_meas <= hi)
        pred_BERs.append(ber_pred)

        # Egyoldali p: H₀: BER_meas >= BER_pred (vagyis NEM jobb)
        k_errors = int(round(ber_meas * n_trials))
        # binomtest: k sikerből n kísérletből, H0: p=ber_pred
        res = binomtest(k_errors, n_trials, ber_pred, alternative='less')
        p_better.append(float(res.pvalue))

    rejected = holm_correct(p_better, alpha)
    n_anomaly = sum(rejected)

    verdict = "TEXTBOOK"
    if n_anomaly >= 2:
        verdict = "ANOMALY"
    elif not all(in_band):
        verdict = "ANOMALY"

    return {
        "verdict":    verdict,
        "delays_tau": delays_tau,
        "pred_BERs":  pred_BERs,
        "meas_BERs":  measured_BERs,
        "in_band":    in_band,
        "p_better":   p_better,
        "holm_rejected": rejected,
        "n_anomaly":  n_anomaly,
    }


# ─── 2a fázis: regeneráció energiaegyezés ────────────────────────────────────

def judge_phase2a(measured_power_w: float, calib: CalibResult,
                  target_amp: float, tol_pct: float = 15.0) -> dict:
    """
    Tankönyvi jóslat: P_mech = ω₀ · E / Q
    E ∝ A²  →  P_pred = ω₀ · (target_amp / A₀)² · E₀ / Q
    ahol E₀ = ½·A₀² (önkényes egységek, csak az arány számít)
    """
    omega0  = 2 * np.pi * calib.f0
    E0      = 0.5 * calib.A0 ** 2
    E_tgt   = E0 * (target_amp / calib.A0) ** 2
    P_pred  = omega0 * E_tgt / calib.Q

    err_pct = abs(measured_power_w - P_pred) / P_pred * 100.0
    verdict = "TEXTBOOK" if err_pct <= tol_pct else "ANOMALY"

    return {
        "verdict":         verdict,
        "P_pred_w":        P_pred,
        "P_meas_w":        measured_power_w,
        "err_pct":         err_pct,
        "tol_pct":         tol_pct,
    }


# ─── 2b fázis: nyitható Q arány ──────────────────────────────────────────────

def judge_phase2b(R_measured: float, Q_open: float, Q_closed: float,
                  R_control: float, min_ratio: float = 10.0) -> dict:
    """
    Tankönyvi jóslat: R ≈ Q_closed / Q_open
    """
    R_pred = Q_closed / Q_open
    # R legalább 10x nagyobb a kontrollnál, és a jóslat sávján belül
    ratio_ok     = R_measured >= min_ratio * R_control
    pred_lo      = R_pred * 0.5
    pred_hi      = R_pred * 2.0
    in_pred_band = pred_lo <= R_measured <= pred_hi

    if ratio_ok and in_pred_band:
        verdict = "TEXTBOOK"
    elif R_measured > R_pred * 2.0 and ratio_ok:
        verdict = "ANOMALY"
    else:
        verdict = "FAIL"

    return {
        "verdict":     verdict,
        "R_pred":      R_pred,
        "R_measured":  R_measured,
        "R_control":   R_control,
        "ratio_ok":    ratio_ok,
        "in_pred_band": in_pred_band,
    }


# ─── 3. fázis: nem-markovi visszafolyás ──────────────────────────────────────

def blp_measure(D_t: np.ndarray, dt: float) -> float:
    """
    Breuer–Laine–Piilo mérték klasszikus megfelelője:
    Σ max(0, dD/dt) · dt a pozitív meredekségű szakaszokon.
    """
    dD = np.diff(D_t)
    return float(np.sum(dD[dD > 0]) * dt)


def predict_D_two_osc(times: np.ndarray, f0_sys: float, Q_sys: float,
                       f0_env: float, Q_env: float,
                       kappa: float, A0: float) -> np.ndarray:
    """
    Két csatolt csillapított oszcillátor analitikus közelítése D(t)-re.
    (Rövid csatolás esetén perturbatív, csak kvalitatív jóslat.)
    """
    omega_s = 2 * np.pi * f0_sys
    omega_e = 2 * np.pi * f0_env
    gamma_s = omega_s / Q_sys
    gamma_e = omega_e / Q_env
    # Csatolt amplitúdó (perturbatív, első rend)
    delta_omega = omega_e - omega_s
    D = A0 * np.exp(-0.5 * gamma_s * times) * np.abs(
        np.cos(delta_omega * times / 2) +
        (kappa / (2 * delta_omega + 1e-9)) * np.sin(delta_omega * times)
    )
    return D


def judge_phase3(D_meas: np.ndarray, D_pred: np.ndarray,
                 dt: float, D_control: np.ndarray) -> dict:
    """
    TEXTBOOK: visszafolyás mértéke és időzítése a jóslaton belül,
              kontroll nulla körüli.
    ANOMALY:  visszafolyás nagyobb / hosszabb, mint a csatolt egyenletek megengednek.
    """
    blp_meas    = blp_measure(D_meas, dt)
    blp_pred    = blp_measure(D_pred, dt)
    blp_ctrl    = blp_measure(D_control, dt)

    # Kontroll: várhatóan ~0 (monoton csökkenés)
    ctrl_ok = blp_ctrl < 0.05 * blp_pred if blp_pred > 0 else blp_ctrl < 1e-6

    # Jóslat sáv: ±50% (perturbatív közelítés miatt széles)
    in_band = 0.0 <= blp_meas <= 1.5 * blp_pred

    if in_band and ctrl_ok:
        verdict = "TEXTBOOK"
    elif blp_meas > 1.5 * blp_pred:
        verdict = "ANOMALY"
    else:
        verdict = "ANOMALY"

    return {
        "verdict":   verdict,
        "blp_meas":  blp_meas,
        "blp_pred":  blp_pred,
        "blp_ctrl":  blp_ctrl,
        "ctrl_ok":   ctrl_ok,
        "in_band":   in_band,
    }


# ─── 4. fázis: tartalom szerinti felidézés ───────────────────────────────────

def judge_phase4(acc_hebb: float, acc_w0: float, acc_wrand: float,
                 n_trials: int, alpha: float = 0.01) -> dict:
    """
    Hebb-csatolás pontossága szignifikánsan nagyobb mindkét kontrollnál.
    Egyoldali binomiális teszt.
    """
    def one_sided_p(acc_a: float, acc_b: float, n: int) -> float:
        # H0: acc_a <= acc_b, egyoldali
        k = int(round(acc_a * n))
        res = binomtest(k, n, acc_b, alternative='greater')
        return float(res.pvalue)

    p_vs_w0   = one_sided_p(acc_hebb, acc_w0,   n_trials)
    p_vs_wrand = one_sided_p(acc_hebb, acc_wrand, n_trials)

    sig_vs_w0   = p_vs_w0   < alpha
    sig_vs_wrand = p_vs_wrand < alpha
    success      = sig_vs_w0 and sig_vs_wrand

    return {
        "verdict":       "TEXTBOOK" if success else "FAIL",
        "acc_hebb":      acc_hebb,
        "acc_w0":        acc_w0,
        "acc_wrand":     acc_wrand,
        "p_vs_w0":       p_vs_w0,
        "p_vs_wrand":    p_vs_wrand,
        "sig_vs_w0":     sig_vs_w0,
        "sig_vs_wrand":  sig_vs_wrand,
    }


# ─── Összefoglaló riport ──────────────────────────────────────────────────────

def print_verdict(phase: str, result: dict):
    v = result.get("verdict", "?")
    icon = {"TEXTBOOK": "✅", "ANOMALY": "🔴", "INVALID": "⛔", "FAIL": "❌"}.get(v, "❓")
    print(f"\n{'─'*60}")
    print(f"  {icon}  {phase}: {v}")
    for k, val in result.items():
        if k != "verdict":
            print(f"     {k}: {val}")
    print(f"{'─'*60}")
