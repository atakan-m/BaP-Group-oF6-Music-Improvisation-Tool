import numpy as np
import sounddevice as sd
from scipy.signal import firwin

# -----------------------Configuration-----------------------
SAMPLE_RATE = 44100 # In Hertz.
FFT_SIZE = 2048 # analyze time so around 46.4 ms of audio at a time.
# HOP_SIZE = 1024 # how far we slide for next analysis, every 23.2 ms.
SILENCE_THRESH = 0.008 # threshold of vol
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

# ----------------------Detection Class-----------------
class PureArrayDetector:
    def __init__(self):
        self.audio_buffer = np.zeros(FFT_SIZE, dtype=np.float32) 
        self.last_midi = None 
        # self.candidate_midi = None 
        # self.candidate_count = 0 
        self.prev_energy = 0.0
        # self.onset = False
        self.melody = []
        self.last_note = None
        self.keyboard = np.zeros(88)

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

    def callback(self, indata, frames, time_info, status):
        
        # Noise Cancel and Update the Audio Buffer
        self.audio_buffer = np.convolve(indata[:, 0], bandpass, "same")    
        
        # Check for onset (change to spectral flux later)
        energy = np.sum(self.audio_buffer ** 2)
        
        rms = np.sqrt(energy / FFT_SIZE)
        
        if rms < SILENCE_THRESH: 
            if self.last_midi is not None: 
                # Print Silence alongside an empty array
                print(f"{'Silence':<7} : {[0] * 88}", flush=True)
                self.last_midi = None     
                self
            return
        
        onset = energy > self.prev_energy * 1.5 #onset if energy is jumped by 50%
        

        #Apply Hann to reduce edge artifacts
        spectrum = np.abs(np.fft.rfft(self.audio_buffer * WINDOW))
        
        #Find Onset
        mask = (FREQS >= LO) & (FREQS <= HI)
        freq_of_max = FREQS[mask][np.argmax(spectrum[mask])]
        
        note_num = self.freq_to_midi(freq_of_max)
        
        note = self.midi_to_name(note_num)
        
        if onset or note != self.last_note:
            self.melody.append(note)
            self.last_note = note
            self.last_midi = note_num
            if 21 <= note_num <= 108: 
                self.keyboard[note_num - 21] = 1
            
            marker = "* " if onset else "  "  # * marks a detected onset
            #print(f"{marker}{freq_of_max:.1f} Hz  ->  {note}   |  {' '.join(self.melody)}")
        else:
            self.keyboard = np.zeros(88)
        

#------Main Loop------
if __name__ == "__main__":
    detector = PureArrayDetector()
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                        blocksize=FFT_SIZE, callback=detector.callback):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            pass