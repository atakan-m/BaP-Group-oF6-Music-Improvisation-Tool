import numpy as np
import sys
import HW.buttons_ml as buttons
import ML.generate as gen
import HW.display_ml as display

def main():
    chord_prog_2 = []
    chord_prog = buttons.main()
    for chord in chord_prog:
        chord_prog_2.append((chord,4))
    print(chord_prog_2)
    notes = gen.generate_music(gen.LSTMmodel, gen.chord_to_id, chord_prog_2*4, temperature=0.8)
    print(notes[:10])
    sheet = display.make_sheet(notes, 36)
    display.display_start(sheet)
    

main()