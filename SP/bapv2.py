import numpy as np
import sounddevice as sd

RATE = 44100
BLOCK = 4096
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
melody = []
last_note = None
prev_energy = 0.0  # energy of the previous block, used for onset detection

def callback(indata, frames, time, status):
    global last_note, prev_energy

    audio = indata[:, 0]

    if audio.max() < 0.01:  # skip silence
        prev_energy = 0.0
        return

    # onset detection: compare current block energy to previous block
    energy = np.sum(audio ** 2)
    onset = energy > prev_energy * 1.5  # onset if energy jumped by 50%
    prev_energy = energy

    # apply Hann window to reduce edge artifacts, then run FFT
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / RATE)

    # find the loudest frequency in the melody range
    mask = (freqs >= 60) & (freqs <= 1000)
    freq = freqs[mask][np.argmax(spectrum[mask])]

    # convert Hz to the nearest MIDI note, then to a note name
    midi = round(69 + 12 * np.log2(freq / 440))
    note = NOTES[midi % 12] + str(midi // 12 - 1)

    # log note on onset or note change
    if onset or note != last_note:
        melody.append(note)
        last_note = note
        marker = "* " if onset else "  "  # * marks a detected onset
        print(f"{marker}{freq:.1f} Hz  ->  {note}   |  {' '.join(melody)}")

with sd.InputStream(samplerate=RATE, channels=1, callback=callback, blocksize=BLOCK):
    print("Play a melody... (Ctrl+C to stop)  * = onset\n")
    try:
        while True: sd.sleep(100)
    except KeyboardInterrupt:
        print(f"\n{' '.join(melody)}")