import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

# ─────────────────────────── Configuration ───────────────────────────
SAMPLE_RATE    = 44100
FFT_SIZE       = 4096      # ~10.8 Hz/bin — good frequency resolution
HOP_SIZE       = 512*2       # ~11.6 ms/callback — 16 callbacks per 16th note @ 80 BPM
A4_FREQ        = 440.0
SILENCE_THRESH = 0.015     # RMS gate; raise by 0.005 if sustain/pedal sticks

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Search range: A0 (27.5 Hz) to C8 (4186 Hz).
# c3 (130) and b5 (987)
# Raise LO_HZ toward ~65 (C2) if low-frequency room rumble causes false onsets.
LO_HZ, HI_HZ  = 127, 1017*3

HPS_HARMONICS    = 5     # 5 keeps the valid HPS region above HI_HZ; 6 would cut off the top
CONFIRM_FRAMES   = 2     # frames a note must be stable before we emit it (~23 ms @ 512 hop)
FLUX_HISTORY     = 22    # ~0.25 s of history for the adaptive onset threshold
FLUX_MULT        = 2   # onset fires when flux > FLUX_MULT × median(recent flux)
REFRACTORY_FRAMES = 4    # ~46 ms lockout after an onset — stops one attack double-firing

# ─────────────────────── Pre-computed globals ─────────────────────────
WINDOW  = np.hanning(FFT_SIZE)
FREQS   = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)

# ---- IIR bandpass (stateful across callbacks, no block-boundary clicks) ----
_sos   = butter(4, [LO_HZ, HI_HZ], btype="bandpass", fs=SAMPLE_RATE, output="sos")

# ---- Bin indices that bracket our search range -------------------------
LO_BIN = int(np.searchsorted(FREQS, LO_HZ))
HI_BIN = int(np.searchsorted(FREQS, HI_HZ))


# ─────────────────────────── Helpers ─────────────────────────────────
def freq_to_midi(freq: float) -> int:
    return round(12.0 * np.log2((freq + 1e-60) / A4_FREQ) + 69)

def midi_to_freq(midinote: float) -> float:
    return 440 * 2.0**((midinote - 69) / 12.0)

def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}"

def parabolic_interp(spectrum: np.ndarray, idx: int) -> float:
    """Sub-bin peak refinement. Returns fractional bin index."""
    idx = int(np.clip(idx, 1, len(spectrum) - 2))
    a, b, g = spectrum[idx-1], spectrum[idx], spectrum[idx+1]
    denom = a - 2*b + g
    if abs(denom) < 1e-10:
        return float(idx)
    return idx + 0.5 * (a - g) / denom

def hps(mag_full: np.ndarray, harmonics: int = HPS_HARMONICS) -> np.ndarray:
    """
    Harmonic Product Spectrum on the FULL rfft magnitude array.
    hps[i] = product of mag_full[i], mag_full[2i], …, mag_full[harmonics·i].
    Valid up to len(mag_full)//harmonics.
    """
    max_len = len(mag_full) // harmonics
    product = mag_full[:max_len].copy()
    for h in range(2, harmonics + 1):
        downsampled = mag_full[::h][:max_len]
        product *= downsampled
    return product


# ──────────────────────────── Detector ───────────────────────────────
class PureArrayDetector:
    def __init__(self):
        self.buf           = np.zeros(FFT_SIZE, dtype=np.float32)
        self.filter_zi     = np.zeros((_sos.shape[0], 2))

        self.prev_mag_norm = np.zeros(FFT_SIZE // 2 + 1)
        self.flux_hist     = np.zeros(FLUX_HISTORY)
        self.timeout       = 0

        self.last_midi     = None
        self.cand_midi     = None
        self.cand_count    = 0

        self.melody        = np.full(16, "", dtype="<U7")
        self.keyboard      = np.zeros(88)

    # ── filter ───────────────────────────────────────────────────────
    def _filter(self, block: np.ndarray) -> np.ndarray:
        out, self.filter_zi = sosfilt(_sos, block, zi=self.filter_zi)
        return out

    # ── onset ─────────────────────────────────────────────────────────
    def _update_onset(self, mag_norm: np.ndarray) -> bool:
        """
        Adaptive-median spectral flux with a refractory lockout.
        - Median threshold auto-scales to the recent signal, so it works
          across dynamic levels without a hand-tuned prominence value.
        - After an onset fires, REFRACTORY_FRAMES suppress further triggers
          so a single percussive attack cannot emit multiple events.
        """
        flux = float(np.sum(np.maximum(mag_norm - self.prev_mag_norm, 0.0)))
        self.prev_mag_norm = mag_norm
        self.flux_hist     = np.roll(self.flux_hist, -1)
        self.flux_hist[-1] = flux

        # Refractory lockout after a recent onset
        if self.timeout > 0:
            self.timeout -= 1
            return False

        nonzero = self.flux_hist[self.flux_hist > 0]
        if len(nonzero) < 3:
            return False
        
        median = float(np.median(nonzero))
        if flux > FLUX_MULT * median:# and flux - median > 5:
            self.timeout = REFRACTORY_FRAMES
            return True
        return False

    # ── pitch ─────────────────────────────────────────────────────────
    def _detect_pitch(self, mag_full: np.ndarray) -> float | None:
        """
        1. Run HPS on the full spectrum.
        2. Constrain the search to [LO_BIN, HI_BIN] after HPS.
        3. Parabolic interpolation on HPS values → fractional bin.
        4. Convert fractional bin → Hz via FREQS.
        5. Octave correction: if the half-frequency bin dominates, go one octave down.
        """
        hps_full = hps(mag_full)

        lo = max(LO_BIN, 1)
        hi = min(HI_BIN, len(hps_full) - 1)
        if lo >= hi:
            return None

        search_region = hps_full[lo:hi]
        local_peak    = int(np.argmax(search_region))
        global_peak   = local_peak + lo

        refined_bin = np.clip(
            parabolic_interp(hps_full, global_peak), 0, len(FREQS) - 1
        )

        # Map fractional bin → Hz (FREQS is linear so lerp is exact)
        lo_b = np.clip(int(np.floor(refined_bin)), 0, len(FREQS) - 1)
        hi_b = np.clip(min(lo_b + 1, len(FREQS) - 1), 0, len(FREQS) - 1)
        frac = refined_bin - lo_b
        freq_hz = FREQS[lo_b] * (1 - frac) + FREQS[hi_b] * frac

        midi = freq_to_midi(freq_hz)
        if not (21 <= midi <= 108):
            return None

        # ── Octave correction ──────────────────────────────────────────
        # HPS can lock onto the first sub-harmonic (one octave too high).
        # If the half-frequency bin dominates, prefer the octave down.
        peak_bin = int(round(refined_bin))
        half_bin = peak_bin // 2
        if half_bin >= 1 and mag_full[half_bin] > 0.6 * mag_full[peak_bin]:
            half_midi = freq_to_midi(freq_hz / 2)
            if 21 <= half_midi <= 108:
                freq_hz = freq_hz / 2
        # ──────────────────────────────────────────────────────────────

        return freq_hz

    # ── callback ──────────────────────────────────────────────────────
    def callback(self, indata, frames, time_info, status):
        # Mono input — take the single channel
        block = indata[:, 0].astype(np.float64)

        # Silence gate — reset state and bail when the room is quiet
        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms < SILENCE_THRESH:
            self.last_midi  = None
            self.cand_midi  = None
            self.cand_count = 0
            
            self.flux_hist     = np.roll(self.flux_hist, -1)
            self.flux_hist[-1] = 1e-20
            
            self.timeout -= 1
            return

        # Filter → ring buffer
        filtered = self._filter(block)
        self.buf[:-HOP_SIZE]  = self.buf[HOP_SIZE:]
        self.buf[-HOP_SIZE:] = filtered

        # FFT — full magnitude
        mag_full = np.abs(np.fft.rfft(self.buf * WINDOW))

        # Normalised magnitude for onset (loudness-independent)
        mag_norm = mag_full / (mag_full.max() + 1e-60)
        onset    = self._update_onset(mag_norm)

        # Pitch
        freq_hz = self._detect_pitch(mag_full)
        if freq_hz is None:
            return

        midi = freq_to_midi(freq_hz)
        note = midi_to_name(midi)

        # Debounce — note must be stable across CONFIRM_FRAMES
        if midi == self.cand_midi:
            self.cand_count += 1
        else:
            self.cand_midi  = midi
            self.cand_count = 1
        if self.cand_count < CONFIRM_FRAMES:
            return

        # Emit only on confirmed onset — spectral flux is the sole trigger.
        if onset:
            if self.last_midi is not None and 21 <= self.last_midi <= 108:
                self.keyboard[self.last_midi - 21] = 0
            self.last_midi = midi

            self.melody[:-1] = self.melody[1:]
            self.melody[-1] = note
            if 21 <= midi <= 108:
                self.keyboard[midi - 21] = 1

            print(f"* {freq_hz:.2f} Hz  →  {note:<4}  "
                  f"MIDI {midi:3d}  |  {' '.join(self.melody)}")


# ────────────────────────────── Main ─────────────────────────────────
def main():
    det = PureArrayDetector()
    print(f"Listening …  FFT={FFT_SIZE}  hop={HOP_SIZE}  "
          f"HPS harmonics={HPS_HARMONICS}  Ctrl-C to stop\n")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=HOP_SIZE, dtype="float32",
                        callback=det.callback):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()