import numpy as np
import sys
# sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//ML')
# sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//HW')
# sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//SP')
import HW.Buttons_v2 as buttons
import ML.generate as gen

def main():

    chord_prog = buttons.main()
    print(chord_prog)

while True:
    main()