"""
plan.py — Sorsolt bitek, minták, próbasorrend
Máté + Maya, 2026-11-11

A sorsolt értékek a mérés előtt befagyasztott seedből jönnek.
A sorsolt fájl elkülönített, lezárt JSON — a dekóder csak a mérés után láthatja.
"""

import numpy as np
import json
import hashlib
import time
from pathlib import Path
from typing import List, Tuple


class TrialPlan:
    """
    Egyetlen fázis próbatervét tárolja.
    
    A tényleges bitek / minták a sealed_path-ban vannak,
    amelyet a decode() metódus nyit fel a mérés lezárása után.
    """
    def __init__(self, seed: int, phase: str):
        self.seed  = seed
        self.phase = phase
        self.rng   = np.random.default_rng(seed)
        self._sealed: dict = {}

    # ── 1. fázis ──────────────────────────────────────────────────────────────

    def phase1_plan(self, delays_tau: List[float],
                    trials_per_delay: int) -> List[Tuple[float, int, int]]:
        """
        Visszaad: [(delay_tau_frac, trial_idx, bit), ...]
        Véletlen sorrendben, bit-ek elrejtve a sealed dict-ben.
        """
        entries = []
        bits_map = {}
        trial_idx = 0
        for d in delays_tau:
            bits = self.rng.integers(0, 2, size=trials_per_delay).tolist()
            for b in bits:
                key = f"p1_{trial_idx}"
                bits_map[key] = int(b)
                entries.append((d, trial_idx, key))
                trial_idx += 1
        # Véletlen sorrend
        order = self.rng.permutation(len(entries)).tolist()
        self._sealed["phase1_bits"] = bits_map
        return [(entries[i][0], entries[i][1], entries[i][2]) for i in order]

    # ── 2. fázis ──────────────────────────────────────────────────────────────

    def phase2_plan(self, target_amps: List[float],
                    runs_per_amp: int) -> List[Tuple[float, int, str]]:
        """
        Visszaad: [(amplitude, run_idx, key), ...]
        """
        entries = []
        bits_map = {}
        run_idx = 0
        for amp in target_amps:
            bits = self.rng.integers(0, 2, size=runs_per_amp).tolist()
            for b in bits:
                key = f"p2_{run_idx}"
                bits_map[key] = int(b)
                entries.append((amp, run_idx, key))
                run_idx += 1
        order = self.rng.permutation(len(entries)).tolist()
        self._sealed["phase2_bits"] = bits_map
        return [(entries[i][0], entries[i][1], entries[i][2]) for i in order]

    # ── 3. fázis ──────────────────────────────────────────────────────────────

    def phase3_plan(self, trials: int) -> List[Tuple[int, str]]:
        """
        Visszaad: [(bit, key), ...] — sorsolt bitek a nem-markovi teszthez
        """
        bits_map = {}
        entries = []
        bits = self.rng.integers(0, 2, size=trials).tolist()
        for i, b in enumerate(bits):
            key = f"p3_{i}"
            bits_map[key] = int(b)
            entries.append((int(b), key))
        order = self.rng.permutation(len(entries)).tolist()
        self._sealed["phase3_bits"] = bits_map
        return [entries[i] for i in order]

    # ── 4. fázis ──────────────────────────────────────────────────────────────

    def phase4_plan(self, n_patterns: int = 2,
                    trials: int = 200,
                    control_trials: int = 200) -> dict:
        """
        Generál két Hebb-mintát (±1, 4 bit), sorsolt felidézési próbákat,
        és kontroll próbákat.
        
        Visszaad dict:
          patterns: (n_patterns, 4) int tömb ±1
          recall_trials: [(pattern_idx, corrupted_bit_idx, key), ...]
          control_trials_w0: [(pattern_idx, corrupted_bit_idx, key), ...]  — W=0
          control_trials_wrand: [(pattern_idx, corrupted_bit_idx, key), ...] — kevert W
        """
        # Véletlenszerű minták (ortogonalitásra törekedve)
        patterns = self.rng.choice([-1, 1], size=(n_patterns, 4))
        self._sealed["phase4_patterns"] = patterns.tolist()

        def gen_trials(label: str, n: int):
            entries = []
            answers = {}
            pat_idxs = self.rng.integers(0, n_patterns, size=n)
            bit_idxs = self.rng.integers(0, 4, size=n)
            for i, (pi, bi) in enumerate(zip(pat_idxs, bit_idxs)):
                key = f"{label}_{i}"
                answers[key] = {"pattern_idx": int(pi), "corrupt_bit": int(bi)}
                entries.append((int(pi), int(bi), key))
            self._sealed[f"{label}_answers"] = answers
            order = self.rng.permutation(n).tolist()
            return [entries[j] for j in order]

        return {
            "patterns":       patterns.tolist(),
            "recall":         gen_trials("p4_recall", trials),
            "ctrl_w0":        gen_trials("p4_ctrl_w0", control_trials),
            "ctrl_wrand":     gen_trials("p4_ctrl_wrand", control_trials),
        }

    # ── Lezárás és megnyitás ───────────────────────────────────────────────────

    def seal(self, path: str) -> str:
        """
        A sorsolt értékeket lezárt JSON fájlba írja.
        Visszaad: SHA-256 hash (a preregistrációba kerül).
        """
        blob = json.dumps({
            "seed":   self.seed,
            "phase":  self.phase,
            "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data":   self._sealed
        }, indent=2, sort_keys=True)
        sha = hashlib.sha256(blob.encode()).hexdigest()
        Path(path).write_text(blob, encoding="utf-8")
        print(f"[plan] Lezárva: {path}")
        print(f"[plan] SHA-256: {sha}")
        return sha

    @staticmethod
    def unseal(path: str, expected_sha: str) -> dict:
        """
        Csak a mérés lezárása után nyitható meg.
        Ellenőrzi a hash-t, visszaad a 'data' dict-et.
        """
        blob = Path(path).read_text(encoding="utf-8")
        computed = hashlib.sha256(blob.encode()).hexdigest()
        if computed != expected_sha:
            raise ValueError(
                f"Sealed plan hash mismatch!\n"
                f"  elvárt:  {expected_sha}\n"
                f"  kapott:  {computed}\n"
                "A fájl módosítva lett — az eredmény érvénytelen."
            )
        return json.loads(blob)["data"]


def generate_all_plans(seed: int, cfg, out_dir: str) -> dict:
    """
    Legenerálja az összes fázis tervét, lezárja, és visszaad
    egy dict-et {phase: sha256} mapping-gal.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    # 1. fázis
    p1 = TrialPlan(seed ^ 0x1, "phase1")
    p1.phase1_plan(cfg.phase1_delays_tau_frac, cfg.phase1_trials_per_delay)
    hashes["phase1"] = p1.seal(str(out / "sealed_phase1.json"))

    # 2. fázis
    p2 = TrialPlan(seed ^ 0x2, "phase2")
    p2.phase2_plan(cfg.phase2_target_amplitudes, cfg.phase2_runs_per_amp)
    hashes["phase2"] = p2.seal(str(out / "sealed_phase2.json"))

    # 3. fázis
    p3 = TrialPlan(seed ^ 0x3, "phase3")
    p3.phase3_plan(200)
    hashes["phase3"] = p3.seal(str(out / "sealed_phase3.json"))

    # 4. fázis
    p4 = TrialPlan(seed ^ 0x4, "phase4")
    p4.phase4_plan(cfg.phase4_n_patterns, cfg.phase4_trials, cfg.phase4_control_trials)
    hashes["phase4"] = p4.seal(str(out / "sealed_phase4.json"))

    # Hash manifeszt
    manifest = json.dumps(hashes, indent=2)
    (out / "plan_manifest.json").write_text(manifest)
    print("[plan] Manifeszt mentve:", out / "plan_manifest.json")
    return hashes
