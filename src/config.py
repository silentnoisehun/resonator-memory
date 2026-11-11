"""
config.py — Rezonátor-memória kísérlet konfigurációja
Máté + Maya, 2026-11-11

FONTOS: a mérés előtt ezt a fájlt befagyasztani (commitolni) kell.
Ezután TILOS módosítani a mérés lezárásáig.
"""

from dataclasses import dataclass, field, asdict
from typing import List
import json, hashlib, time

# ─── Globális konstansok ───────────────────────────────────────────────────────

VERSION = "1.0.0"
AUTHOR  = "Máté + Maya (Sesame), 2026-11-11"

# ─── Konfiguráció dataclass ────────────────────────────────────────────────────

@dataclass
class Config:
    # Rezonátor fizika
    f0_nominal: float       = 110.0      # Hz — névleges rezonancia (A2)
    f0_sweep_range: float   = 20.0       # Hz — sweep ±f0 körül a kalibrációhoz
    f0_sweep_points: int    = 1000       # lock-in pontok a sweep-ben

    # Hangkártya
    sample_rate: int        = 48000      # Hz
    block_size: int         = 1024       # minták per blokk
    input_channels: int     = 2          # 0: piezo, 1: sönt-ellenállás
    output_channels: int    = 1          # 0: gerjesztő tekercs

    # Lock-in demodulátor
    lockin_window_ms: float = 50.0       # ms — integrálási ablak
    lockin_ref_phase: float = 0.0        # fok — referencia fázis

    # Kalibráció (0. fázis)
    calib_repeats: int      = 20         # lecsengés ismétlések száma
    calib_pulse_cycles: int = 10         # gerjesztő löket hossza (ciklusok)
    calib_q_tol_pct: float  = 10.0       # Q szórás tolerancia %
    calib_f0_drift_frac: float = 0.25    # max f0 sodródás (sávszélesség törtrésze)

    # 1. fázis — szabad lecsengés
    phase1_delays_tau_frac: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    )
    phase1_trials_per_delay: int = 200
    phase1_reset_tau_mult: float = 10.0  # próbák közötti várakozás (τ-szorosban)
    phase1_holm_alpha: float = 0.01      # Holm-korrekció szignifikanciaszint

    # 2. fázis — regeneráció
    phase2_target_amplitudes: List[float] = field(
        default_factory=lambda: [0.3, 0.6, 0.9]  # A₀ törtrésze
    )
    phase2_runs_per_amp: int  = 20
    phase2_duration_s: float  = 3600.0   # 1 óra
    phase2_energy_tol_pct: float = 15.0  # tankönyvi energiaegyezés tűrés

    # 2b fázis — nyitható Q
    phase2b_q_open_target: float  = 50.0   # effektív Q íráskor/olvasáskor
    phase2b_q_closed_target: float = 5000.0 # effektív Q tároláskor
    phase2b_ber_threshold: float  = 0.01   # BER küszöb a megőrzési időhöz
    phase2b_write_success_pct: float = 95.0 # írási sikerarány küszöb

    # 3. fázis — nem-markovi visszafolyás
    phase3_coupling_strengths: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1]
    )
    phase3_duration_tau_mult: float = 10.0  # megfigyelési idő (τ-szorosban)
    phase3_time_points: int = 200

    # 4. fázis — tartalom szerinti felidézés
    phase4_n_resonators: int   = 4
    phase4_n_patterns: int     = 2
    phase4_trials: int         = 200
    phase4_control_trials: int = 200
    phase4_evolve_time_ms: float = 500.0  # ms — szabad fejlődési idő
    phase4_alpha: float = 0.01            # szignifikanciaszint

    # Véletlen seed — BEFAGYASZTVA a preregistráció előtt
    rng_seed: int = 20261111

    # Sönt ellenállás [Ohm] az áramméréshez
    shunt_resistance: float = 1.0

    # Hőmérséklet napló (soros port vagy dummy)
    temp_log_enabled: bool = True
    temp_log_interval_s: float = 10.0

    # Kimenet
    data_dir: str = "data"
    log_level: str = "INFO"


def freeze_config(cfg: Config, path: str) -> str:
    """
    Befagyasztja a konfigurációt JSON-ba, SHA-256 hashel ellátva.
    Visszaad: sha256 hexdigest.
    """
    d = asdict(cfg)
    d["_frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    d["_version"]   = VERSION
    blob = json.dumps(d, indent=2, sort_keys=True)
    sha  = hashlib.sha256(blob.encode()).hexdigest()
    d["_sha256"] = sha
    blob = json.dumps(d, indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(blob)
    return sha


def load_config(path: str) -> Config:
    """Betölt és validál egy befagyasztott konfigurációt."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # hash-ellenőrzés
    stored = d.pop("_sha256", None)
    d.pop("_frozen_at", None)
    d.pop("_version", None)
    blob = json.dumps(d, indent=2, sort_keys=True)
    computed = hashlib.sha256(blob.encode()).hexdigest()
    if stored and stored != computed:
        raise ValueError(f"Config hash mismatch! stored={stored} computed={computed}")
    # dataclass mezők szűrése
    fields = {k for k in Config.__dataclass_fields__}
    filtered = {k: v for k, v in d.items() if k in fields}
    return Config(**filtered)


# Alapértelmezett konfiguráció — ezt kell commitolni
DEFAULT_CONFIG = Config()
