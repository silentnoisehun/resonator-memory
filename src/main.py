"""
main.py — Rezonátor-memória kísérlet főprogram
Máté + Maya, 2026-11-11

Parancsok:
  python main.py dryrun           — szimulációs száraz futtatás
  python main.py plan             — sorsolt tervek generálása és lezárása
  python main.py phase0           — kalibráció (valódi HW vagy szimuláció)
  python main.py phase1           — 1. fázis mérés
  python main.py phase2a          — 2a fázis mérés (regeneráció)
  python main.py phase2b          — 2b fázis mérés (nyitható Q)
  python main.py phase3           — 3. fázis mérés (nem-markovi)
  python main.py phase4           — 4. fázis mérés (asszociatív felidézés)
  python main.py all              — teljes szekvencia (szimuláción)

Kapcsolók:
  --sim         szimulációt használ HW helyett (alapértelmezett ha nincs HW)
  --config PATH befagyasztott config JSON
  --out   PATH  kimenet könyvtár (alapértelmezett: data/)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config import Config, DEFAULT_CONFIG, freeze_config, load_config
from dryrun import run_all as dryrun_all
from plan import generate_all_plans, TrialPlan
from analysis import (
    CalibResult, check_calib, judge_phase1, judge_phase2a,
    judge_phase2b, judge_phase3, judge_phase4, print_verdict,
    predict_D_two_osc
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")


# ─── Hardver / szimulációs injektálás ────────────────────────────────────────

def get_hw(cfg: Config, use_sim: bool):
    """Visszaad egy HWInterface vagy SimHardware példányt."""
    if use_sim:
        import numpy as np
        from sim import OscParams, SingleOscillator, SimHardware
        rng = np.random.default_rng(cfg.rng_seed)
        p   = OscParams(f0=cfg.f0_nominal, Q=500.0, noise_sigma=0.02,
                        sample_rate=cfg.sample_rate)
        osc = SingleOscillator(p, rng)
        return SimHardware(osc, cfg.lockin_window_ms)
    else:
        from hw import HWConfig, HWInterface
        hw_cfg = HWConfig(
            f0=cfg.f0_nominal,
            sample_rate=cfg.sample_rate,
            shunt_resistance=cfg.shunt_resistance,
            lockin_window_ms=cfg.lockin_window_ms,
        )
        iface = HWInterface(hw_cfg)
        iface.open()
        return iface


# ─── 0. fázis ─────────────────────────────────────────────────────────────────

def cmd_phase0(cfg: Config, hw, out_dir: Path) -> CalibResult:
    logger.info("0. fázis: kalibráció indul")
    import numpy as np
    from scipy.optimize import curve_fit

    sample_rate = cfg.sample_rate
    window      = int(cfg.lockin_window_ms * 1e-3 * sample_rate)
    tau_guess   = cfg.phase2b_q_closed_target / (np.pi * cfg.f0_nominal)

    taus, amps = [], []
    for rep in range(cfg.calib_repeats):
        hw.write_bit(0, cfg.calib_pulse_cycles)
        I_vals = []
        t_vals = []
        # ~5 τ lecsengés mintavételezése
        n_steps = 20
        for s in range(n_steps):
            hw.wait_tau(0.25, tau_guess)
            I = hw.read_I()
            I_vals.append(abs(I))
            t_vals.append((s + 1) * 0.25 * tau_guess)
        try:
            popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau),
                                 t_vals, I_vals, p0=[I_vals[0], tau_guess])
            taus.append(popt[1])
            amps.append(popt[0])
        except Exception:
            logger.warning("rep %d: curve_fit sikertelen, kihagyva", rep)

    if len(taus) < 10:
        logger.error("Nem elég sikeres lecsengés-illesztés (%d < 10)", len(taus))
        sys.exit("INVALID kalibráció")

    tau_mean = float(np.mean(taus))
    tau_std  = float(np.std(taus))
    Q_mean   = np.pi * cfg.f0_nominal * tau_mean
    Q_std    = np.pi * cfg.f0_nominal * tau_std
    A0_mean  = float(np.mean(amps))

    # Zaj mérése
    sigma_vals = []
    for _ in range(5):
        I = hw.read_I()
        sigma_vals.append(abs(I))
    sigma = float(np.std(sigma_vals)) + 1e-6

    calib = CalibResult(
        f0=cfg.f0_nominal, Q=Q_mean, tau=tau_mean,
        sigma=sigma, A0=A0_mean,
        Q_std=Q_std, f0_drift=0.0, valid=True
    )
    verdict = check_calib(calib, cfg)
    logger.info("Kalibráció ítélet: %s  Q=%.1f±%.1f  τ=%.4fs",
                verdict, Q_mean, Q_std, tau_mean)

    out = {"verdict": verdict, **calib.__dict__}
    (out_dir / "calib_result.json").write_text(json.dumps(out, indent=2))

    if verdict == "INVALID":
        sys.exit("0. fázis INVALID — mérés nem folytatható.")
    return calib


# ─── 1. fázis ─────────────────────────────────────────────────────────────────

def cmd_phase1(cfg: Config, hw, calib: CalibResult,
               sealed_path: str, expected_sha: str, out_dir: Path):
    logger.info("1. fázis: szabad lecsengés BER indul")
    import numpy as np

    # Sealed plan megnyitása (a mérés után)
    plan_data = TrialPlan.unseal(sealed_path, expected_sha)
    bits_map  = plan_data["phase1_bits"]

    p1 = TrialPlan(cfg.rng_seed ^ 0x1, "phase1")
    trials = p1.phase1_plan(cfg.phase1_delays_tau_frac, cfg.phase1_trials_per_delay)

    results_by_delay = {d: {"errors": 0, "total": 0}
                        for d in cfg.phase1_delays_tau_frac}

    for delay_frac, trial_idx, key in trials:
        bit = bits_map[key]
        hw.write_bit(bit, cfg.calib_pulse_cycles)
        hw.wait_tau(delay_frac, calib.tau)
        I = hw.read_I()
        read_bit = 0 if I >= 0 else 1
        if read_bit != bit:
            results_by_delay[delay_frac]["errors"] += 1
        results_by_delay[delay_frac]["total"] += 1
        # Reset
        hw.wait_tau(cfg.phase1_reset_tau_mult, calib.tau)

    meas_BERs = [
        results_by_delay[d]["errors"] / results_by_delay[d]["total"]
        for d in cfg.phase1_delays_tau_frac
    ]
    result = judge_phase1(
        cfg.phase1_delays_tau_frac, meas_BERs,
        cfg.phase1_trials_per_delay, calib, cfg.phase1_holm_alpha
    )
    print_verdict("1. fázis", result)
    (out_dir / "phase1_result.json").write_text(json.dumps(result, indent=2))
    return result


# ─── Argument parser ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Rezonátor-memória kísérlet")
    p.add_argument("command", choices=[
        "dryrun", "plan", "phase0", "phase1",
        "phase2a", "phase2b", "phase3", "phase4", "all"
    ])
    p.add_argument("--sim",    action="store_true", help="Szimulációt használ HW helyett")
    p.add_argument("--config", default=None,        help="Befagyasztott config JSON")
    p.add_argument("--out",    default="data",      help="Kimenet könyvtár")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config betöltés
    if args.config:
        cfg = load_config(args.config)
        logger.info("Config betöltve: %s", args.config)
    else:
        cfg = DEFAULT_CONFIG
        # Első futásnál befagyaszt
        cfg_path = out_dir / "config_frozen.json"
        if not cfg_path.exists():
            sha = freeze_config(cfg, str(cfg_path))
            logger.info("Config befagyasztva: %s  SHA=%s", cfg_path, sha[:12])

    # ── Parancsok ─────────────────────────────────────────────────────────────

    if args.command == "dryrun":
        ok = dryrun_all(cfg)
        sys.exit(0 if ok else 1)

    elif args.command == "plan":
        hashes = generate_all_plans(cfg.rng_seed, cfg, str(out_dir / "sealed"))
        (out_dir / "plan_hashes.json").write_text(json.dumps(hashes, indent=2))
        logger.info("Sealed tervek és hash-ek mentve.")

    elif args.command == "phase0":
        hw = get_hw(cfg, args.sim)
        try:
            calib = cmd_phase0(cfg, hw, out_dir)
        finally:
            hw.close()

    elif args.command == "phase1":
        calib_path = out_dir / "calib_result.json"
        if not calib_path.exists():
            sys.exit("Először futtasd: phase0")
        raw   = json.loads(calib_path.read_text())
        calib = CalibResult(**{k: raw[k] for k in CalibResult.__dataclass_fields__})
        hashes = json.loads((out_dir / "plan_hashes.json").read_text())
        hw = get_hw(cfg, args.sim)
        try:
            cmd_phase1(cfg, hw, calib,
                       str(out_dir / "sealed" / "sealed_phase1.json"),
                       hashes["phase1"], out_dir)
        finally:
            hw.close()

    elif args.command == "all":
        logger.info("Teljes szekvencia szimulációval...")
        ok = dryrun_all(cfg)
        sys.exit(0 if ok else 1)

    else:
        logger.warning("%s: részletes implementáció a következő fázisokban", args.command)


if __name__ == "__main__":
    main()
