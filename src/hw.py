"""
hw.py — Valódi hangkártya I/O és lock-in interfész
Máté + Maya, 2026-11-11

Függőségek: sounddevice, numpy, scipy
Telepítés: pip install sounddevice numpy scipy

FONTOS: Ha nincs hangkártya, a hw.py helyett sim.SimHardware-t kell injektálni.
"""

import numpy as np
import threading
import time
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    logger.warning("sounddevice nem elérhető — csak szimuláció futtatható.")


@dataclass
class HWConfig:
    f0: float            # Hz — kalibrált rezonancia
    sample_rate: int     # Hz
    input_device: Optional[int]  = None   # None = default
    output_device: Optional[int] = None
    block_size: int              = 1024
    shunt_resistance: float      = 1.0    # Ohm
    lockin_window_ms: float      = 50.0
    ref_phase_deg: float         = 0.0


class HWInterface:
    """
    Valódi hangkártya-alapú rezonátor interfész.
    
    Csatornakiosztás (bemeneti):
      ch0 → piezo hangszedő (rezonátor mozgás)
      ch1 → sönt-ellenálláson mért feszültség (gerjesztő áram)
    
    Csatornakiosztás (kimeneti):
      ch0 → gerjesztő tekercs / hangszóró
    """
    def __init__(self, cfg: HWConfig):
        if not SD_AVAILABLE:
            raise RuntimeError("sounddevice nem elérhető")
        self.cfg = cfg
        self.omega0 = 2 * np.pi * cfg.f0
        self.dt     = 1.0 / cfg.sample_rate
        self.window_samples = int(cfg.lockin_window_ms * 1e-3 * cfg.sample_rate)
        self._ring_buf = np.zeros((cfg.sample_rate * 2, 2))  # 2s ring
        self._ring_ptr = 0
        self._ring_lock = threading.Lock()
        self._stream: Optional[sd.Stream] = None
        self._output_buf: Optional[np.ndarray] = None

    # ── Inicializálás ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Megnyitja a hangkártya stream-et."""
        self._stream = sd.Stream(
            samplerate=self.cfg.sample_rate,
            blocksize=self.cfg.block_size,
            device=(self.cfg.input_device, self.cfg.output_device),
            channels=(2, 1),
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()
        logger.info("HWInterface megnyílt: f0=%.2f Hz, sr=%d Hz",
                    self.cfg.f0, self.cfg.sample_rate)

    def close(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, outdata, frames, time_info, status):
        """sounddevice callback — főszál helyett audio-szálban fut."""
        if status:
            logger.warning("sd status: %s", status)
        # Ring buffer frissítés
        with self._ring_lock:
            n = len(indata)
            end = self._ring_ptr + n
            L = len(self._ring_buf)
            if end <= L:
                self._ring_buf[self._ring_ptr:end] = indata
            else:
                first = L - self._ring_ptr
                self._ring_buf[self._ring_ptr:] = indata[:first]
                self._ring_buf[:end - L]        = indata[first:]
            self._ring_ptr = end % L
        # Kimenet
        if self._output_buf is not None and len(self._output_buf) >= frames:
            outdata[:, 0] = self._output_buf[:frames]
            self._output_buf = self._output_buf[frames:]
            if len(self._output_buf) == 0:
                self._output_buf = None
        else:
            outdata.fill(0)

    # ── Lock-in olvasás ───────────────────────────────────────────────────────

    def _get_ring_seg(self, n_samples: int) -> np.ndarray:
        """Az utolsó n_samples mintát adja a ring bufferből."""
        with self._ring_lock:
            L = len(self._ring_buf)
            start = (self._ring_ptr - n_samples) % L
            if start < self._ring_ptr:
                return self._ring_buf[start:self._ring_ptr].copy()
            else:
                return np.concatenate([
                    self._ring_buf[start:],
                    self._ring_buf[:self._ring_ptr]
                ])

    def read_IQ(self) -> Tuple[float, float]:
        """
        Digitális lock-in demoduláció az utolsó window_ms adatból.
        Visszaad: (I, Q) — piezo ch0-ból.
        """
        seg = self._get_ring_seg(self.window_samples)
        x   = seg[:, 0]  # piezo
        n   = len(x)
        t   = np.arange(n) * self.dt
        phi = np.deg2rad(self.cfg.ref_phase_deg)
        ref_i = np.sin(self.omega0 * t + phi)
        ref_q = np.cos(self.omega0 * t + phi)
        I = 2.0 * float(np.mean(x * ref_i))
        Q = 2.0 * float(np.mean(x * ref_q))
        return I, Q

    def read_I(self) -> float:
        return self.read_IQ()[0]

    def read_power_w(self) -> float:
        """
        Felvett teljesítmény a sönt-ellenállásból.
        P = V_shunt² / R_shunt (átlagolva az ablakra)
        """
        seg = self._get_ring_seg(self.window_samples)
        v_shunt = seg[:, 1]
        return float(np.mean(v_shunt ** 2)) / self.cfg.shunt_resistance

    # ── Gerjesztés ────────────────────────────────────────────────────────────

    def _make_pulse(self, phase_deg: float, n_cycles: int,
                    amplitude: float = 0.5) -> np.ndarray:
        n = int(n_cycles / self.cfg.f0 * self.cfg.sample_rate)
        t = np.arange(n) * self.dt
        phi = np.deg2rad(phase_deg)
        # Hanning ablak a kattanás elkerülésére
        win = np.hanning(n)
        return (amplitude * win * np.sin(self.omega0 * t + phi)).astype(np.float32)

    def write_bit(self, bit: int, n_cycles: int = 10,
                  amplitude: float = 0.5) -> None:
        """Bit írása: 0 → 0° fázis, 1 → 180° fázis."""
        phase = 0.0 if bit == 0 else 180.0
        pulse = self._make_pulse(phase, n_cycles, amplitude)
        self._output_buf = pulse
        # Megvárjuk, amíg a löket lefut
        wait_s = len(pulse) / self.cfg.sample_rate + 0.05
        time.sleep(wait_s)

    def active_damp(self, duration_s: float, gain: float = -0.5) -> None:
        """
        Aktív csillapítás: negatív visszacsatolás a kiolvasott I komponensből.
        Csak demonstrációs célra (valódi implementáció DSP-ben kell).
        """
        end = time.time() + duration_s
        while time.time() < end:
            I, Q = self.read_IQ()
            amp  = np.hypot(I, Q)
            # Visszacsatoló löket: 90° fáziseltolással az energia elvenni
            pulse = self._make_pulse(
                phase_deg=np.rad2deg(np.arctan2(Q, I)) + 90.0,
                n_cycles=1,
                amplitude=float(np.clip(gain * amp, -0.3, 0.0))
            )
            self._output_buf = pulse
            time.sleep(max(0, len(pulse) / self.cfg.sample_rate))

    def wait_tau(self, n_tau: float, tau: float) -> None:
        """Vár n_tau × τ másodpercet."""
        time.sleep(n_tau * tau)

    def temp_celsius(self) -> float:
        """
        Hőmérséklet olvasás — soros portról (placeholder).
        Valódi implementációnál: pyserial-lal kell felváltani.
        """
        return float("nan")
