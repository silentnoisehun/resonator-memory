"""
sim.py — Zajos csillapított oszcillátor szimuláció
Ugyanazt az I/O felületet adja, mint a hw.py.

Modellek:
  - Egy oszcillátor (1–2. fázis)
  - Két csatolt oszcillátor (3. fázis)
  - Négy csatolt oszcillátor (4. fázis)

A mozgásegyenlet (Euler-Maruyama):
  ẍ + (ω₀/Q)ẋ + ω₀²x = F(t)/m + σ_noise·ξ(t)

ahol ξ(t) fehér zaj, F(t) a gerjesztő erő.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class OscParams:
    f0: float         # Hz
    Q: float          # minőségi tényező
    A0: float = 1.0   # kezdeti amplitúdó [önkényes egység]
    noise_sigma: float = 0.01  # zaj szint (A₀ egységekben)
    sample_rate: int = 48000   # Hz


class SingleOscillator:
    """
    Egy csillapított, gerjesztett harmonikus oszcillátor.
    Lock-in demoduláción átmenve I/Q szignálokat ad.
    """
    def __init__(self, params: OscParams, rng: np.random.Generator):
        self.p   = params
        self.rng = rng
        self.omega0 = 2 * np.pi * params.f0
        self.dt = 1.0 / params.sample_rate
        # Állapot: [x, v]
        self.state = np.array([0.0, 0.0])

    def reset(self):
        self.state[:] = 0.0

    def excite_pulse(self, phase_deg: float, n_cycles: int) -> None:
        """
        Rövid szinuszos gerjesztő löket, adott fázissal.
        phase_deg: 0 → '0' bit, 180 → '1' bit
        """
        omega = self.omega0
        n_samples = int(n_cycles / self.p.f0 * self.p.sample_rate)
        phi = np.deg2rad(phase_deg)
        for i in range(n_samples):
            t = i * self.dt
            F = self.p.A0 * np.sin(omega * t + phi)
            self._step(F)

    def _step(self, F: float) -> None:
        x, v = self.state
        noise = self.rng.normal(0, self.p.noise_sigma / np.sqrt(self.dt))
        a = F - (self.omega0 / self.p.Q) * v - self.omega0**2 * x + noise
        v_new = v + a * self.dt
        x_new = x + v_new * self.dt
        self.state = np.array([x_new, v_new])

    def run_free(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Szabad lecsengés: n_samples mintát generál.
        Visszaad: (time_arr, x_arr)
        """
        xs = np.zeros(n_samples)
        for i in range(n_samples):
            self._step(0.0)
            xs[i] = self.state[0]
        ts = np.arange(n_samples) * self.dt
        return ts, xs

    def run_with_drive(self, drive: np.ndarray) -> np.ndarray:
        """Adott gerjesztő tömbbel futtat, visszaad x tömböt."""
        xs = np.zeros(len(drive))
        for i, F in enumerate(drive):
            self._step(F)
            xs[i] = self.state[0]
        return xs

    def lockin_demodulate(self, x_arr: np.ndarray,
                           window_samples: int,
                           ref_phase_deg: float = 0.0) -> Tuple[float, float]:
        """
        Digitális lock-in demoduláció az utolsó window_samples mintán.
        Visszaad: (I, Q)
        """
        seg = x_arr[-window_samples:]
        n   = len(seg)
        t   = np.arange(n) * self.dt
        phi = np.deg2rad(ref_phase_deg)
        ref_i = np.sin(self.omega0 * t + phi)
        ref_q = np.cos(self.omega0 * t + phi)
        I = 2.0 * np.mean(seg * ref_i)
        Q = 2.0 * np.mean(seg * ref_q)
        return float(I), float(Q)


class TwoOscillator:
    """
    Két csatolt oszcillátor: rendszer (alacsony Q) + környezet (magas Q).
    Csatolás: κ · (x_env - x_sys) erő mindkét irányban.
    """
    def __init__(self, sys_params: OscParams, env_params: OscParams,
                 coupling: float, rng: np.random.Generator):
        self.sys = SingleOscillator(sys_params, rng)
        self.env = SingleOscillator(env_params, rng)
        self.kappa = coupling
        self.dt = sys_params.dt if hasattr(sys_params, 'dt') else 1.0 / sys_params.sample_rate

    def reset(self):
        self.sys.reset()
        self.env.reset()

    def excite_sys(self, phase_deg: float, n_cycles: int):
        self.sys.excite_pulse(phase_deg, n_cycles)

    def step(self) -> Tuple[float, float]:
        """Egy szimulációs lépés. Visszaad (x_sys, x_env)."""
        x_s, v_s = self.sys.state
        x_e, v_e = self.env.state

        F_coupling_s = self.kappa * (x_e - x_s)
        F_coupling_e = self.kappa * (x_s - x_e)

        self.sys._step(F_coupling_s)
        self.env._step(F_coupling_e)
        return self.sys.state[0], self.env.state[0]

    def run(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        xs_sys = np.zeros(n_samples)
        xs_env = np.zeros(n_samples)
        for i in range(n_samples):
            xs_sys[i], xs_env[i] = self.step()
        return xs_sys, xs_env


class FourOscillator:
    """
    Négy oszcillátor Hopfield-stílusú szoftver-csatolással.
    A csatolási mátrix W-t a kísérlet állítja be (Hebb).
    Az állapot: négy rezonátor fázisa.
    """
    def __init__(self, params_list: List[OscParams], rng: np.random.Generator):
        assert len(params_list) == 4
        self.oscs = [SingleOscillator(p, rng) for p in params_list]
        self.W = np.zeros((4, 4))
        self.dt = params_list[0].dt if hasattr(params_list[0], 'dt') else \
                  1.0 / params_list[0].sample_rate
        self.window_samples = int(50e-3 * params_list[0].sample_rate)  # 50 ms

    def set_hebb_weights(self, patterns: np.ndarray):
        """
        patterns: (n_patterns, 4) tömb ±1 értékekkel.
        W = Σ ξξᵀ / n_patterns, diagonális = 0
        """
        n = patterns.shape[0]
        self.W = (patterns.T @ patterns) / n
        np.fill_diagonal(self.W, 0.0)

    def reset(self):
        for o in self.oscs:
            o.reset()

    def set_state(self, phases_deg: np.ndarray, n_cycles: int = 10):
        """Beállítja az oszcillátorok állapotát a megadott fázisokra."""
        for osc, ph in zip(self.oscs, phases_deg):
            osc.reset()
            osc.excite_pulse(float(ph), n_cycles)

    def read_phases(self, ref_phase_deg: float = 0.0) -> np.ndarray:
        """Kiolvassa a négy oszcillátor I komponensét (fázis előjele)."""
        result = []
        for osc in self.oscs:
            _, xs = osc.run_free(self.window_samples)
            I, _ = osc.lockin_demodulate(xs, self.window_samples, ref_phase_deg)
            result.append(np.sign(I) if abs(I) > 1e-9 else 0.0)
        return np.array(result)

    def hopfield_step(self, alpha: float = 0.1):
        """
        Egy Hopfield-frissítési lépés:
        h_i = Σ_j W_ij · sign(I_j)
        Az oszcillátor i-t h_i előjelének megfelelő fázissal gerjesztjük.
        alpha: gerjesztés erőssége (A₀ törtrésze)
        """
        phases = self.read_phases()
        h = self.W @ phases
        for i, osc in enumerate(self.oscs):
            drive_phase = 0.0 if h[i] >= 0 else 180.0
            force = alpha * osc.p.A0
            # Kis löket a Hopfield irányban
            n = int(5 / osc.p.f0 * osc.p.sample_rate)
            phi = np.deg2rad(drive_phase)
            omega = osc.omega0
            for k in range(n):
                t = k * osc.dt
                F = force * np.sin(omega * t + phi)
                osc._step(F)

    def run_recall(self, n_steps: int, alpha: float = 0.1) -> np.ndarray:
        """
        n_steps Hopfield-lépés után kiolvassa a végső fázisokat.
        """
        for _ in range(n_steps):
            self.hopfield_step(alpha)
        return self.read_phases()


# ─── Szimulált hardver-interfész (hw.py kompatibilis API) ─────────────────────

class SimHardware:
    """
    Virtuális hangkártya / lock-in — ugyanazt az API-t adja, mint HWInterface.
    Használat: a hw.py HWInterface helyett injektálható.
    """
    def __init__(self, osc: SingleOscillator, window_ms: float = 50.0):
        self.osc = osc
        self.window_ms = window_ms
        self.window_samples = int(window_ms * 1e-3 * osc.p.sample_rate)
        self._last_power_w = 0.0

    def write_bit(self, bit: int, n_cycles: int = 10) -> None:
        phase = 0.0 if bit == 0 else 180.0
        self.osc.excite_pulse(phase, n_cycles)

    def read_I(self) -> float:
        """I komponens kiolvasása (lock-in), jelenlegi állapotból."""
        _, xs = self.osc.run_free(self.window_samples)
        I, _ = self.osc.lockin_demodulate(xs, self.window_samples)
        return I

    def read_IQ(self) -> Tuple[float, float]:
        _, xs = self.osc.run_free(self.window_samples)
        return self.osc.lockin_demodulate(xs, self.window_samples)

    def wait_tau(self, n_tau: float, tau: float) -> None:
        n_samples = int(n_tau * tau * self.osc.p.sample_rate)
        self.osc.run_free(n_samples)

    def read_power_w(self) -> float:
        """Szimulált teljesítmény: P = ω₀ · E / Q"""
        x, v = self.osc.state
        E = 0.5 * (v**2 + self.osc.omega0**2 * x**2)
        return self.osc.omega0 * E / self.osc.p.Q

    def temp_celsius(self) -> float:
        return 22.0 + self.osc.rng.normal(0, 0.1)

    def close(self):
        pass
