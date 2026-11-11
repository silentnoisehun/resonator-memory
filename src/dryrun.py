"""
dryrun.py — Minden fázis száraz futtatása szimulációval
Máté + Maya, 2026-11-11

Elvárás: minden fázis TEXTBOOK ítéletet ad.
Pozitív kontroll: mesterségesen javított szimulációnak ANOMALY-t kell adnia.
"""

import numpy as np
import logging
from config import Config, DEFAULT_CONFIG
from sim import OscParams, SingleOscillator, TwoOscillator, FourOscillator, SimHardware
from analysis import (
    CalibResult, check_calib,
    predict_BER, judge_phase1,
    judge_phase2a, judge_phase2b,
    predict_D_two_osc, judge_phase3,
    judge_phase4, print_verdict
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_osc(cfg: Config, Q: float = None, rng_seed: int = 42) -> SingleOscillator:
    rng = np.random.default_rng(rng_seed)
    p = OscParams(
        f0=cfg.f0_nominal,
        Q=Q or 500.0,
        A0=1.0,
        noise_sigma=0.02,
        sample_rate=cfg.sample_rate,
    )
    return SingleOscillator(p, rng)


# ─── 0. fázis szimulált kalibráció ───────────────────────────────────────────

def sim_calibrate(cfg: Config, osc: SingleOscillator) -> CalibResult:
    """Szimulált lecsengésből kalibráció."""
    tau_true = osc.p.Q / (np.pi * osc.p.f0)
    n_samples = int(5 * tau_true * cfg.sample_rate)
    window    = int(cfg.lockin_window_ms * 1e-3 * cfg.sample_rate)

    amplitudes = []
    for _ in range(cfg.calib_repeats):
        osc.reset()
        osc.excite_pulse(0.0, cfg.calib_pulse_cycles)
        _, xs = osc.run_free(n_samples)
        I, _ = osc.lockin_demodulate(xs, window)
        amplitudes.append(abs(I))

    # τ a lecsengésből (exponenciális illesztés helyett közelítés)
    tau_est = tau_true * (1 + osc.rng.normal(0, 0.02))
    Q_est   = np.pi * osc.p.f0 * tau_est

    # zaj mérése (gerjesztés nélkül)
    osc.reset()
    _, xs_noise = osc.run_free(window * 4)
    I_noise, _ = osc.lockin_demodulate(xs_noise, window)
    sigma = abs(I_noise) + 0.001  # kis bias a nulla elkerülésére

    return CalibResult(
        f0       = osc.p.f0,
        Q        = Q_est,
        tau      = tau_est,
        sigma    = max(sigma, osc.p.noise_sigma * 0.5),
        A0       = float(np.mean(amplitudes)),
        Q_std    = float(np.std(amplitudes) / np.mean(amplitudes) * Q_est),
        f0_drift = 0.01,  # szimulált: stabil
        valid    = True
    )


# ─── Dryrun fázisok ───────────────────────────────────────────────────────────

def dryrun_phase0(cfg: Config):
    print("\n══ DRYRUN 0. fázis: kalibráció ══")
    osc   = make_osc(cfg)
    calib = sim_calibrate(cfg, osc)
    res   = check_calib(calib, cfg)
    print(f"  f0={calib.f0:.2f} Hz  Q={calib.Q:.1f}  τ={calib.tau:.3f}s  σ={calib.sigma:.4f}")
    print(f"  Ítélet: {res}")
    assert res == "VALID", "0. fázis INVALID — szimulációs hiba!"
    return calib


def dryrun_phase1(cfg: Config, calib: CalibResult):
    print("\n══ DRYRUN 1. fázis: szabad lecsengés BER ══")
    osc     = make_osc(cfg)
    hw      = SimHardware(osc, cfg.lockin_window_ms)
    rng     = np.random.default_rng(cfg.rng_seed ^ 0x1)
    window  = int(cfg.lockin_window_ms * 1e-3 * cfg.sample_rate)

    meas_BERs = []
    for delay_frac in cfg.phase1_delays_tau_frac:
        delay_s = delay_frac * calib.tau
        errors  = 0
        for _ in range(cfg.phase1_trials_per_delay):
            bit = int(rng.integers(0, 2))
            osc.reset()
            hw.write_bit(bit, cfg.calib_pulse_cycles)
            hw.wait_tau(delay_frac, calib.tau)
            I = hw.read_I()
            read_bit = 0 if I >= 0 else 1
            if read_bit != bit:
                errors += 1
        meas_BERs.append(errors / cfg.phase1_trials_per_delay)

    result = judge_phase1(
        cfg.phase1_delays_tau_frac, meas_BERs,
        cfg.phase1_trials_per_delay, calib, cfg.phase1_holm_alpha
    )
    print_verdict("1. fázis", result)
    return result


def dryrun_phase2a(cfg: Config, calib: CalibResult):
    print("\n══ DRYRUN 2a fázis: regeneráció ══")
    osc = make_osc(cfg)

    results = []
    for amp in cfg.phase2_target_amplitudes:
        osc.reset()
        osc.excite_pulse(0.0, cfg.calib_pulse_cycles)
        # Szimulált regeneráció: a valódi teljesítmény a lecsengési veszteségből
        x, v = osc.state
        E_tgt = 0.5 * (amp * calib.A0) ** 2
        P_sim = osc.omega0 * E_tgt / osc.p.Q
        # Kis mérési zaj
        P_meas = P_sim * (1 + osc.rng.normal(0, 0.05))
        res = judge_phase2a(P_meas, calib, amp * calib.A0, cfg.phase2_energy_tol_pct)
        print_verdict(f"2a fázis (amp={amp:.1f})", res)
        results.append(res)
    return results


def dryrun_phase2b(cfg: Config, calib: CalibResult):
    print("\n══ DRYRUN 2b fázis: nyitható Q ══")
    Q_open   = cfg.phase2b_q_open_target
    Q_closed = cfg.phase2b_q_closed_target

    # R = megőrzési_idő / írási_idő ~ Q_closed / Q_open
    R_pred    = Q_closed / Q_open
    R_meas    = R_pred * np.random.default_rng(42).uniform(0.6, 1.0)
    R_control = 1.2  # fix Q-nál várható

    result = judge_phase2b(R_meas, Q_open, Q_closed, R_control)
    print_verdict("2b fázis", result)
    return result


def dryrun_phase3(cfg: Config, calib: CalibResult):
    print("\n══ DRYRUN 3. fázis: nem-markovi visszafolyás ══")
    rng = np.random.default_rng(cfg.rng_seed ^ 0x3)

    f0_sys, Q_sys = calib.f0,       200.0
    f0_env, Q_env = calib.f0 * 1.05, 2000.0
    kappa = 0.05 * 2 * np.pi * f0_sys

    dt      = 1.0 / cfg.sample_rate
    n_pts   = cfg.phase3_time_points
    tau_sys = Q_sys / (np.pi * f0_sys)
    t_max   = cfg.phase3_duration_tau_mult * tau_sys
    times   = np.linspace(0, t_max, n_pts)

    D_pred = predict_D_two_osc(times, f0_sys, Q_sys, f0_env, Q_env, kappa, calib.A0)
    D_meas = D_pred * (1 + rng.normal(0, 0.05, n_pts))  # 5% zaj
    D_ctrl = calib.A0 * np.exp(-times / tau_sys)         # monoton csökkenés

    result = judge_phase3(D_meas, D_pred, times[1]-times[0], D_ctrl)
    print_verdict("3. fázis", result)
    return result


def dryrun_phase4(cfg: Config):
    print("\n══ DRYRUN 4. fázis: tartalom szerinti felidézés ══")
    rng = np.random.default_rng(cfg.rng_seed ^ 0x4)

    params_list = [
        OscParams(f0=cfg.f0_nominal + i * 2.0, Q=500.0,
                  noise_sigma=0.02, sample_rate=cfg.sample_rate)
        for i in range(4)
    ]
    four = FourOscillator(params_list, rng)

    # Két Hebb-minta
    patterns = np.array([[1, 1, -1, -1], [-1, 1, 1, -1]])
    four.set_hebb_weights(patterns)

    def run_trial(pattern_idx: int, corrupt_bit: int, W_mode: str) -> bool:
        init = patterns[pattern_idx].copy().astype(float)
        init[corrupt_bit] *= -1
        init_phases = np.where(init > 0, 0.0, 180.0)
        four.reset()
        four.set_state(init_phases, n_cycles=5)

        if W_mode == "hebb":
            final = four.run_recall(n_steps=10, alpha=0.15)
        elif W_mode == "w0":
            four.W[:] = 0
            final = four.read_phases()
        else:  # wrand
            W_rand = four.W.copy()
            idx = rng.permutation(4)
            four.W = W_rand[np.ix_(idx, idx)]
            final = four.run_recall(n_steps=10, alpha=0.15)

        return bool(np.all(np.sign(final) == patterns[pattern_idx]))

    n = cfg.phase4_trials
    hebb_ok  = sum(run_trial(rng.integers(0,2), rng.integers(0,4), "hebb")  for _ in range(n))
    w0_ok    = sum(run_trial(rng.integers(0,2), rng.integers(0,4), "w0")    for _ in range(n))
    wrand_ok = sum(run_trial(rng.integers(0,2), rng.integers(0,4), "wrand") for _ in range(n))

    result = judge_phase4(hebb_ok/n, w0_ok/n, wrand_ok/n, n, cfg.phase4_alpha)
    print_verdict("4. fázis", result)
    return result


# ─── Pozitív kontroll: mesterségesen javított szimulációnak ANOMALY-t kell adnia

def positive_control_phase1(cfg: Config, calib: CalibResult):
    """
    Ha a BER mesterségesen 0 (mindig helyes), az analysis ANOMALY-t kell adjon.
    """
    print("\n══ POZITÍV KONTROLL (1. fázis) — elvárás: ANOMALY ══")
    meas_BERs = [0.0] * len(cfg.phase1_delays_tau_frac)  # tökéletes, fizikailag lehetetlen
    result = judge_phase1(
        cfg.phase1_delays_tau_frac, meas_BERs,
        cfg.phase1_trials_per_delay, calib, cfg.phase1_holm_alpha
    )
    print_verdict("Pozitív kontroll", result)
    assert result["verdict"] == "ANOMALY", \
        "HIBA: a pozitív kontroll nem adott ANOMALY-t — az analysis hibás!"
    print("  ✅ Pozitív kontroll OK: ANOMALY detektálva.")


# ─── Teljes dryrun ────────────────────────────────────────────────────────────

def run_all(cfg: Config = None):
    if cfg is None:
        cfg = DEFAULT_CONFIG

    print("=" * 60)
    print("  REZONÁTOR-MEMÓRIA DRYRUN")
    print("  Minden fázis szimulációval, elvárt ítélet: TEXTBOOK")
    print("=" * 60)

    calib = dryrun_phase0(cfg)
    r1    = dryrun_phase1(cfg, calib)
    r2a   = dryrun_phase2a(cfg, calib)
    r2b   = dryrun_phase2b(cfg, calib)
    r3    = dryrun_phase3(cfg, calib)
    r4    = dryrun_phase4(cfg)
    positive_control_phase1(cfg, calib)

    print("\n" + "=" * 60)
    verdicts = {
        "0. fázis (kalib)": "VALID",
        "1. fázis":         r1["verdict"],
        "2a fázis":         r2a[0]["verdict"],
        "2b fázis":         r2b["verdict"],
        "3. fázis":         r3["verdict"],
        "4. fázis":         r4["verdict"],
    }
    all_ok = all(v in ("TEXTBOOK", "VALID") for v in verdicts.values())
    for name, v in verdicts.items():
        icon = "✅" if v in ("TEXTBOOK","VALID") else "❌"
        print(f"  {icon}  {name}: {v}")

    print("=" * 60)
    if all_ok:
        print("  ✅ DRYRUN ÁTMENT — kész az előregisztrációra.")
    else:
        print("  ❌ DRYRUN HIBÁS — az elemzőkódot javítani kell, NEM mérünk!")
    return all_ok


if __name__ == "__main__":
    run_all()
