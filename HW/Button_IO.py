import sys

def main():

    chord_dict = {0: "A",
                  1: "A#",
                  2: "B",
                  3: "C",
                  4: "C#",
                  5: "D",
                  6: "D#",
                  7: "E",
                  8: "F",
                  9: "F#",
                  10: "G",
                  11: "G#"}
    
    mod_dict = {0: "",
                1: "m",
                2: "sus2",
                3: "sus4",
                4: "dim",
                5: "aug",
                6: "7",
                7: "dim7",
                8: "maj7",
                9: "m7",
                10: "aug7",
                11: "6",
                12: "m6",
                13: "9",
                14: "min9"}

    done = False
    confirm = False
    reset = False
    chord_up = False
    chord_down = False
    mod_up = False
    mod_down = False


    current_chord = [0, 0]
    chord_prog = []

    while True:
        x = input("Enter command: ")

        match x:
            case 'w':
                current_chord[0] += 1
                current_chord[0] = current_chord[0] % 12
            case 's':
                current_chord[0] -= 1
                if current_chord[0] < 0:
                    current_chord[0] += 12
                else :
                    current_chord[0] = current_chord[0] % 12
            case '+':
                current_chord[1] += 1
                current_chord[1] = current_chord[1] % 15
            case '-':
                current_chord[1] -= 1
                if current_chord[1] < 0:
                    current_chord[1] += 15
                else :
                    current_chord[1] = current_chord[1] % 15
            case 'c':
                chord_prog.append(chord_dict[current_chord[0]] + mod_dict[current_chord[1]])
                current_chord = [0,0]
            case 'd':
                return chord_prog
            case 'q':
                return
            case _:
                print("Use w/s or +/- or c/d/q")
        print("current chord:", chord_dict[current_chord[0]] + mod_dict[current_chord[1]])
        print("current chord progression:", chord_prog)


print(main())