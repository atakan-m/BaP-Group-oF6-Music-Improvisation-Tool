import numpy as np
import sys
# sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//HW')
# sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//HW')
# sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//SP')

import HW.buttons_ml as buttons
import ML.generate as gen
import HW.display_ml as display

def main():

    chord_prog = buttons.main()
    print(chord_prog)
    notes = gen.generate_music(gen.chord_to_id, chord_prog, temperature=0.8)
    sheet = display.make_sheet(notes, 36)
    display.display_start(sheet)
    

while True:
    main()