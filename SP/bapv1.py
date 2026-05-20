import numpy as np
import sounddevice as sd
from scipy.signal import firwin

# -----------------------Configuration-----------------------
SAMPLE_RATE = 44100 # In Hertz.
FFT_SIZE = 2048 # analyze time so around 46.4 ms of audio at a time.
HOP_SIZE = 1024 # how far we slide for next analysis, every 23.2 ms.
A4_FREQ = 440.0 # anchor music note
SILENCE_THRESH = 0.02

# Note names for string conversion
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ----------------------Pre computed math-----------------
WINDOW = np.hanning(FFT_SIZE) 
FREQS = np.fft.rfftfreq(FFT_SIZE, d=1.0/SAMPLE_RATE) 
LO = 20 
HI = 4200 
bandpass = firwin(100, (LO, HI), pass_zero="bandpass", fs=SAMPLE_RATE)
mask = (FREQS >= LO) & (FREQS <= HI)

# ----------------------Detection Class-----------------
class PureArrayDetector:
    def __init__(self):
        self.audio_buffer = np.zeros(FFT_SIZE, dtype=np.float32) 
        self.last_midi = None 
        # self.candidate_midi = None 
        # self.candidate_count = 0 
        self.prev_energy = 0.0
        # self.onset = False
        self.melody = np.full(10, "", dtype='<U7')
        self.last_note = None
        self.keyboard = np.zeros(88)
        self.spectral_flux = np.zeros(50)

    def freq_to_midi(self, freq):
        """Converts a raw frequency in Hz into a standard MIDI note number."""
        return round(12 * np.log2((freq + 1e-60) / A4_FREQ) + 69)
    
    def midi_to_freq(self, midinote):
        return 440 * 2.0**((midinote - 69) / 12.0)

    def midi_to_name(self, midi_note):
        """Converts a standard MIDI note number into its musical string representation."""
        return f"{NOTE_NAMES[midi_note % 12]}{(midi_note // 12) - 1}"
    
    def spec_diff_H(self, data):
        return (data + np.abs(data)) / 2
    
    def pos_only(self, data):
        return (data + np.abs(data)) / 2

    def compute_flux_pos(self, magnitude):
        diffs = magnitude[1:] - magnitude[:-1]
        return np.sum(self.pos_only(diffs))
    
    def note_equal(self, note1, note2):
        # Check#1 is if has the same note name
        # Check#2 is if note2 has on the sme octave or higher
        if note1 is None or note2 is None:
            return False
        
        if note1[:-1] == note2[:-1] and note2[-1] >= note1[-1]:
            return True
        return False

    def callback(self, indata, frames, time_info, status):
        
        # Noise Cancel and Update the Audio Buffer
        self.audio_buffer = np.roll(self.audio_buffer, -HOP_SIZE) 
        self.audio_buffer[-HOP_SIZE:] = np.convolve(indata[:, 0], bandpass, "same")    
        
        # Check for onset (change to spectral flux later)
        energy = np.sum(self.audio_buffer ** 2)
        
        # Silence Sheck
        rms = np.sqrt(energy / FFT_SIZE)
        if rms < SILENCE_THRESH: 
            if self.last_note is not None: 
                self.last_note = None 
            return
        
        # onset = energy > self.prev_energy * 1.5 #onset if energy is jumped by 50%
        

        #Apply Hann to reduce edge artifacts
        spectrum = np.abs(np.fft.rfft(self.audio_buffer * WINDOW))[mask]
        
        #Find Onset
        self.spectral_flux= np.roll(self.spectral_flux, -1)
        self.spectral_flux[-1] = self.compute_flux_pos(spectrum) / 700000
        
        onset_thresh = self.spectral_flux[-2] * 2
        onset = self.spectral_flux[-1] > onset_thresh
        
        freq_of_max = FREQS[mask][np.argmax(spectrum)]
        
        note_num = self.freq_to_midi(freq_of_max)
        note = self.midi_to_name(note_num)
        
        if onset or not self.note_equal(self.last_note, note):
            self.melody = np.roll(self.melody, 1)
            self.melody[0] = note
            self.last_note = note
            
            if 21 <= note_num <= 108: 
                self.keyboard[note_num - 21] = 1
            
            marker = "* " if onset else "  "  # * marks a detected onset
            print(f"{marker}{freq_of_max:.1f} Hz  ->  {note}   |  {' '.join(self.melody)}")
        else:
            self.keyboard = np.zeros(88)
        

#------Main Loop------
if __name__ == "__main__":
    detector = PureArrayDetector()
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                        blocksize=HOP_SIZE, callback=detector.callback):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            pass