import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt

# ─────────────────────────── Configuration ───────────────────────────
SAMPLE_RATE    = 44100
FFT_SIZE       = 4096      # ~10.8 Hz/bin — good frequency resolution
HOP_SIZE       = 1024      # callback block size 
A4_FREQ        = 440.0
SILENCE_THRESH = 0.005     # RMS gate

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Search range: A0 (27.5 Hz) to C8 (4186 Hz)
LO_HZ, HI_HZ  = 27.5, 4200.0

HPS_HARMONICS  = 5    # how many harmonics to fold down
CONFIRM_FRAMES = 2    # frames a note must be stable before we emit it
FLUX_HISTORY   = 43   # ~1 s of history for adaptive onset threshold
FLUX_MULT      = 1.5  # how many × median = onset

# ─────────────────────── Pre-computed globals ─────────────────────────
WINDOW  = np.hanning(FFT_SIZE)
# Full FFT frequency axis
FREQS   = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)

# ---- IIR bandpass (stateful across callbacks, no block-boundary clicks) ----
_sos   = butter(4, [LO_HZ, HI_HZ], btype="bandpass", fs=SAMPLE_RATE, output="sos")

# ---- Bin indices that bracket our search range -------------------------
# We search the FULL spectrum with HPS, but only inside [LO_HZ, HI_HZ].
# Keep these as integer indices into the raw rfft output.
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
        downsampled = mag_full[::h][:max_len]     # downsample then hard-clip to max_len
        product *= downsampled
    return product                                 # length = len(mag_full)//harmonics


# ──────────────────────────── Detector ───────────────────────────────
class PureArrayDetector:
    def __init__(self):
        self.buf           = np.zeros(FFT_SIZE, dtype=np.float64)
        self.filter_zi     = np.zeros((_sos.shape[0], 2))

        self.prev_mag = np.zeros(FFT_SIZE // 2 + 1)
        self.flux_hist     = np.zeros(FLUX_HISTORY)

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
    def _update_onset(self, mag: np.ndarray) -> bool:
        flux = float(np.sum(np.maximum(mag - self.prev_mag, 0.0)))
        self.prev_mag = mag
        self.flux_hist     = np.roll(self.flux_hist, -1)
        self.flux_hist[-1] = flux
        nonzero = self.flux_hist[self.flux_hist > 0]
        flux_max = np.max(self.flux_hist)
        if (flux - np.median(nonzero)) / flux_max < 0.2:
            return False
        return flux > FLUX_MULT * float(np.median(nonzero))

    # ── pitch ─────────────────────────────────────────────────────────
    def _detect_pitch(self, mag_full: np.ndarray) -> float | None:
        """
        1. Run HPS on the full spectrum (avoids the masked-index bug).
        2. Constrain the search to [LO_BIN, HI_BIN] AFTER HPS.
        3. Parabolic interpolation on HPS values → fractional bin.
        4. Convert fractional bin → Hz via FREQS.
        """
        hps_full = hps(mag_full)          # length = FFT_SIZE/2+1 // HPS_HARMONICS

        # Translate our Hz limits into HPS-array indices.
        # hps_full[i] corresponds to FREQS[i] (same indexing, just shorter array).
        lo = max(LO_BIN, 1)
        hi = min(HI_BIN, len(hps_full) - 1)
        if lo >= hi:
            return None

        search_region   = hps_full[lo:hi]
        local_peak      = int(np.argmax(search_region))
        global_peak     = local_peak + lo          # index into hps_full / FREQS

        refined_bin     = parabolic_interp(hps_full, global_peak)

        # Map fractional bin → Hz (FREQS is linear so lerp is exact)
        lo_b = int(np.floor(refined_bin))
        hi_b = min(lo_b + 1, len(FREQS) - 1)
        frac = refined_bin - lo_b
        freq_hz = FREQS[lo_b] * (1 - frac) + FREQS[hi_b] * frac

        midi = freq_to_midi(freq_hz)
        if not (21 <= midi <= 108):
            return None

        # ── Octave correction ──────────────────────────────────────────
        # HPS commonly locks onto the sub-harmonic (one octave too low).
        # If the octave-up bin has > 30% of the peak's energy, prefer it.
        peak_bin   = int(round(refined_bin))
        double_bin = peak_bin * 2
        if double_bin < len(mag_full):
            if mag_full[double_bin] > 0.3 * mag_full[peak_bin]:
                double_freq = freq_hz * 2
                double_midi = freq_to_midi(double_freq)
                if 21 <= double_midi <= 108:
                    freq_hz = double_freq
        # ──────────────────────────────────────────────────────────────

        return freq_hz

    # ── callback ──────────────────────────────────────────────────────
    def callback(self, indata, frames, time_info, status):
        block = np.mean(indata, axis=1).astype(np.float64)
        
        # Silence gate
        rms = float(np.sqrt(np.mean(block ** 2)))
        # # print(int(rms))
        if rms < SILENCE_THRESH:
            self.last_midi  = None
            self.cand_midi  = None
            self.cand_count = 0
            return

        # Filter → ring buffer
        filtered = self._filter(block)
        self.buf  = np.roll(self.buf, -HOP_SIZE)
        self.buf[-HOP_SIZE:] = filtered

        # FFT — full magnitude (not masked)
        mag_full = np.abs(np.fft.rfft(self.buf * WINDOW))

        # Normalised magnitude for onset (loudness-independent)
        # mag_norm = mag_full / (mag_full.max() + 1e-60)
        onset    = self._update_onset(mag_full)

        # Pitch
        freq_hz = self._detect_pitch(mag_full)
        if freq_hz is None:
            return

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

        # Emit on change or onset
        if midi != self.last_midi or onset:
            if self.last_midi is not None and 21 <= self.last_midi <= 108:
                self.keyboard[self.last_midi - 21] = 0
            self.last_midi = midi

            self.melody    = np.roll(self.melody, -1)
            self.melody[-1] = note
            if 21 <= midi <= 108:
                self.keyboard[midi - 21] = 1

            marker = "* " if onset else "  "
            print(f"{marker}{freq_hz:.2f} Hz  →  {note:<4}  "
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
                sd.sleep(500)
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()  