import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, find_peaks

# ─────────────────────────── Configuration ───────────────────────────
SAMPLE_RATE    = 44100
FFT_SIZE       = 4096      # ~10.8 Hz/bin — good frequency resolution
HOP_SIZE       = 512       # ~11.6 ms/callback — 16 callbacks per 16th note @ 80 BPM
A4_FREQ        = 440.0
SILENCE_THRESH = 0.015      # tuned for 110 dB SPL mic; bump by 0.005 if sustain sticks

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Search range: A0 (27.5 Hz) to C8 (4186 Hz), c2 (65)
LO_HZ, HI_HZ  = 65, 4200.0

HPS_HARMONICS  = 6    # how many harmonics to fold down
CONFIRM_FRAMES = 2    # frames a note must be stable before we emit it (~23 ms @ 512 hop)
FLUX_HISTORY   = 18   # ~0.20 s of history for adaptive onset threshold
FLUX_MULT      = 2  # slightly lower — onset is now sole trigger so must be reliable

# ─────────────────────── Pre-computed globals ─────────────────────────
WINDOW  = np.hanning(FFT_SIZE)
# Full FFT frequency axis
FREQS   = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)

# ---- IIR bandpass (stateful across callbacks, no block-boundary clicks) ----
_sos   = butter(4, [LO_HZ, HI_HZ], btype="bandpass", fs=SAMPLE_RATE, output="sos")

# ---- Bin indices that bracket our search range -------------------------
LO_BIN = int(np.searchsorted(FREQS, LO_HZ))
HI_BIN = int(np.searchsorted(FREQS, HI_HZ))


# ─────────────────────────── Helpers ─────────────────────────────────
def freq_to_midi(freq: float) -> int:
    return round(12.0 * np.log2((freq + 1e-60) / A4_FREQ) + 69)

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
    Returns an array the same length as mag_full where each bin
    [i] = product of mag_full[i*1], mag_full[i*2], …, mag_full[i*harmonics].
    Only valid up to len(mag_full)//harmonics.
    """
    max_len = len(mag_full) // harmonics
    product = mag_full[:max_len].copy()
    for h in range(2, harmonics + 1):
        downsampled = mag_full[::h][:max_len]
        product *= downsampled
    return product  # length = len(mag_full)//harmonics


# ──────────────────────────── Detector ───────────────────────────────
class PureArrayDetector:
    def __init__(self):
        self.buf           = np.zeros(FFT_SIZE, dtype=np.float64)
        self.filter_zi     = np.zeros((_sos.shape[0], 2))

        self.prev_mag_norm = np.zeros(FFT_SIZE // 2 + 1)
        self.flux_hist     = np.zeros(FLUX_HISTORY)
        self.timeout      = 0

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
        flux = float(np.sum(np.maximum(mag_norm - self.prev_mag_norm, 0.0)))
        self.prev_mag_norm = mag_norm
        self.flux_hist     = np.roll(self.flux_hist, -1)
        self.flux_hist[-1] = flux
        nonzero = self.flux_hist[self.flux_hist > 0]

        max_idx = np.argmax(self.flux_hist)
        max_flux = self.flux_hist[max_idx]

        if  self.timeout > 0 or (max_flux - np.median(nonzero)) / max_flux < 0.7:
            self.timeout = np.clip(self.timeout - 1, 0, FLUX_HISTORY)
            return False
        
        peaks, _ = find_peaks(self.flux_hist, distance= FLUX_HISTORY, prominence= 20)
        
        if len(peaks) == 0 or max_idx - 3  > peaks[0] or max_idx + 3  < peaks[0]:
            return False
            
        self.timeout = int(FLUX_HISTORY  * 1.5)
        return True

    # ── pitch ─────────────────────────────────────────────────────────
    def _detect_pitch(self, mag_full: np.ndarray) -> float | None:
        """
        1. Run HPS on the full spectrum.
        2. Constrain the search to [LO_BIN, HI_BIN] after HPS.
        3. Parabolic interpolation on HPS values → fractional bin.
        4. Convert fractional bin → Hz via FREQS.
        5. Octave correction: if the half-frequency bin dominates, go one octave down.
        """
        hps_full = hps(mag_full)  # length = FFT_SIZE/2+1 // HPS_HARMONICS

        lo = max(LO_BIN, 1)
        hi = min(HI_BIN, len(hps_full) - 1)
        if lo >= hi:
            return None

        search_region = hps_full[lo:hi]
        local_peak    = int(np.argmax(search_region))
        global_peak   = local_peak + lo       # index into hps_full / FREQS

        refined_bin   = np.clip(parabolic_interp(hps_full, global_peak), 0, len(FREQS))

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
        # If the half-frequency bin has substantially more energy than the
        # detected peak, prefer the octave down.
        peak_bin = int(round(refined_bin))
        half_bin = peak_bin // 2
        if half_bin >= 1:
            if mag_full[half_bin] > 0.6 * mag_full[peak_bin]:
                half_freq = freq_hz / 2
                half_midi = freq_to_midi(half_freq)
                if 21 <= half_midi <= 108:
                    freq_hz = half_freq
        # ──────────────────────────────────────────────────────────────

        return freq_hz

    # ── callback ──────────────────────────────────────────────────────
    def callback(self, indata, frames, time_info, status):
        block = np.mean(indata, axis=1).astype(np.float64)
        
        # Filter → ring buffer
        filtered = self._filter(block)
        self.buf  = np.roll(self.buf, -HOP_SIZE)
        self.buf[-HOP_SIZE:] = filtered

        # FFT — full magnitude (not masked)
        mag_full = np.abs(np.fft.rfft(self.buf * WINDOW))

        onset    = self._update_onset(mag_full)

        # Pitch
        freq_hz = self._detect_pitch(mag_full)
        
        if freq_hz is None:
            freq_hz = 0
            onset = False

        midi = freq_to_midi(freq_hz)
        note = midi_to_name(midi)

        # Debounce
        if midi == self.cand_midi:
            self.cand_count += 1
        else:
            self.cand_midi  = midi
            self.cand_count = 1
        if self.cand_count < CONFIRM_FRAMES:
            return

        # Emit only on confirmed onset — spectral flux is the sole trigger.
        # midi change alone (e.g. sustain drift) is not enough to emit.
        if onset:
            if self.last_midi is not None and 21 <= self.last_midi <= 108:
                self.keyboard[self.last_midi - 21] = 0
            self.last_midi = midi

            self.melody    = np.roll(self.melody, -1)
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
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=2,
                        blocksize=HOP_SIZE, dtype="float32",
                        callback=det.callback):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()