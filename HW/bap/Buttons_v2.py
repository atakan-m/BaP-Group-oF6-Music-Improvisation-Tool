import sys
import pygame

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


    pygame.init()
    current_chord = [0, 0]
    chord_prog = []
    size = (1920, 1080) # width, height
    flags = pygame.FULLSCREEN
    screen = pygame.display.set_mode(size, flags, vsync=1)
    pygame.display.set_caption("Input")
    WHITE = (255, 255, 255)
    BLACK = (0,0,0)
    fps = 60
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Calibri', 50, True, False)
    font1 = pygame.font.SysFont('Calibri', 20, True, False)

    
    

    if reset == False:
        if confirm == True:
            chord_prog.append(current_chord)
            if done == True:
                pygame.quit()
                return chord_prog
    else:
        current_chord = [0, 0]
        chord_prog = []  
    while not done:
        screen.fill(WHITE)
        
        text = font.render("current chord:" + str(chord_dict[current_chord[0]]) + str(mod_dict[current_chord[1]]),True, BLACK)
        text1 = font.render("current chord progression:" +  str(chord_prog),True, BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:    #so there might be a library that turns gpio inputs into keyboard inputs, which would mean that this could stay as-is :D
                match event.key:
                    case pygame.K_w:
                        current_chord[0] += 1
                        current_chord[0] = current_chord[0] % 12
                    case pygame.K_s:
                        current_chord[0] -= 1
                        if current_chord[0] < 0:
                            current_chord[0] += 12
                        else :
                            current_chord[0] = current_chord[0] % 12
                    case pygame.K_UP:
                        current_chord[1] += 1
                        current_chord[1] = current_chord[1] % 15
                    case pygame.K_DOWN:
                        current_chord[1] -= 1
                        if current_chord[1] < 0:
                            current_chord[1] += 15
                        else :
                            current_chord[1] = current_chord[1] % 15
                    case pygame.K_c:
                        chord_prog.append(chord_dict[current_chord[0]] + mod_dict[current_chord[1]])
                        current_chord = [0,0]
                    case pygame.K_p:
                        chord_prog = ["Am7","Bm7","E9","E9"]
                    case pygame.K_d:
                        pygame.quit()
                        return chord_prog
                    case pygame.K_q:
                        done = True
                    case _:
                        print("Use w/s or up/down or c/d/q")
        #print("current chord:", chord_dict[current_chord[0]] + mod_dict[current_chord[1]])
        #print("current chord progression:", chord_prog)
        text = font.render("current chord:" + str(chord_dict[current_chord[0]]) + str(mod_dict[current_chord[1]]),True, BLACK)
        text1 = font1.render("current chord progression:" +  str(chord_prog),True, BLACK)
        text2 = font1.render("Use the blue buttons and/or green buttons to change the chord, press the yellow button to add to list and the black button to confirm", True, BLACK)
        text3 = font1.render("Red to quit", True, BLACK)
        screen.blit(text, [200, 200])
        screen.blit(text1, [200, 400])
        screen.blit(text2, [200, 600])
        screen.blit(text3, [200, 640])
        pygame.display.flip()
        clock.tick(fps)

if __name__ == "__main__":
    main()