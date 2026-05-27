import pygame
import random
import numpy as np
from collections import deque
import sys 
sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//ML')
sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//SP')
sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//HW//bap')
import generate
import torch
import json
import bapv1
import Buttons_v2
import sounddevice as sd

SAMPLE_RATE = 44100
FFT_SIZE = 2048
WIDTH = 36
HEIGHT = 20
fps = 60 #bpm
bpm = 120
Testing_variable_for_testing = (fps*fps//bpm//4)
detector = bapv1.PureArrayDetector()

hor_pos = [0,0.5,1,1.5,2,3,3.5,4,4.5,5,5.5,6,7,7.5,8,8.5,9,10,10.5,11,11.5,12,12.5,13,14,14.5,15,15.5,16,17,17.5,18,18.5,19,19.5,20]
key_width = [1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1]
buffer_sheet = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         ]

sheet = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ]

col_start = 48
#notes = generate.generate_music(generate.LSTMmodel, generate.chord_to_id, generate.iiVI_long, temperature=0.95)
def note_to_col(note):
    col = note - col_start
    return col

def notes_in_row(notes, num_cols= 36):
    row = [0]*num_cols
    col = note_to_col(notes[0])
    row[col-1] = 1
    return row

def make_sheet(notes, num_cols=36 ):
    sheet = [notes_in_row(note, num_cols)for note in notes]
    return sheet


inputty = set()
if len(bapv1.PureArrayDetector().melody) > 0:
    inputty = {bapv1.PureArrayDetector().melody[0]}

figures = [
        1,2,3,4,5,6,7,8,9,10,11,12,
         13,14,15,16,17,18,19,20,21,22,23,24,
         25,26,27,28,29,30,31,32,33,34,35,36
        ]
Note_list = [
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    ]
x = 0 
def check_note(game, inputty, x):
    setty = set()
    if len(game.figure) != 0:
        for i in game.figure[0].image:
            if i:
                setty.add(i)
            if game.figure[0].y > 15 and inputty.issubset(setty):
                game.figure.popleft()
                return True
            #if game.figure[0].y > 15:

    return None

class Figure:

    def __init__(self, Notes):
        self.x = 0 #note
        self.y = 0
        self.tune = Notes #Name of note

        self._image = [figures[i] if Notes[i] != 0 else '' for i in range(len(Notes))]

        self._note_names = [Note_list[i] if Notes[i] != 0 else '' for i in range(len(Notes))]

    @property
    def image(self):
        return self._image

    @property
    def noteName(self):
        return self._note_names


class Music:
    def __init__(self, height, width):
        self.score = 0
        self.x = 60
        self.y = 80
        self.zoom = 50
        self.figure = deque()
        self.fuck = False
        self.beat_bars = [0]   
        self.height = height
        self.width = width

    def new_figure(self, bar):
            self.figure.append(Figure(bar))
    
    def new_beatbar(self):
        self.beat_bars.append(0)

        if len(self.beat_bars) > 3:
            self.beat_bars.pop(0)
        
    def go_down(self):
        for k in range(len(self.figure)):
            self.figure[k].y += (1/Testing_variable_for_testing)
        for h in range(len(self.beat_bars)):
            self.beat_bars[h] += (1/Testing_variable_for_testing)
            
            
#get the chords and generate notes
chord_prog_2 = []
for chord in Buttons_v2.main():
    chord_prog_2.append((chord,4))
notes = generate.generate_music(generate.LSTMmodel, generate.chord_to_id, chord_prog_2*4, temperature=0.8)

sheet = make_sheet(notes,36)
total_sheet = buffer_sheet + sheet

real_pos = [60 + 50 * hor_pos[p] * 36/21 + 2 for p in range(36)]

# Initialize the game engine
pygame.init()
print(" starting ")

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)

size = (1920, 1080) # width, height
flags = pygame.FULLSCREEN
screen = pygame.display.set_mode(size, flags, vsync=1)
font = pygame.font.SysFont('Calibri', 40, True, False)
font1 = pygame.font.SysFont('Calibri', 50, True, False)
pygame.display.set_caption("Jazz")

# Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()

game = Music(HEIGHT, WIDTH) #height, width
counter = 0
tally = 0
pressing_down = False
sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                blocksize=FFT_SIZE, callback=detector.callback).start()


while not done:
    counter += 1
    if counter > 100000:
        counter = 0

    if counter % (Testing_variable_for_testing) == 0 or pressing_down:
        if tally < len(total_sheet):
            game.new_figure(total_sheet[tally])
            tally += 1
        if len(game.figure) != 0 and game.figure[0].y > 19:
            game.figure.popleft()
        if len(game.figure) != 0 and not any(game.figure[0].image):
            game.figure.popleft()
        
        #if not game.fuck and (game.score - 1) >= 0:
            #game.score -= 1
        game.fuck = False
    
    if counter % (Testing_variable_for_testing*Testing_variable_for_testing) == 0:
        game.new_beatbar()
            

    game.go_down()
    if check_note(game, inputty, game.score) and inputty and counter > 16 :
        game.score += 1
    #else:
        #game.score -= 1
            
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                game.__init__(HEIGHT, WIDTH)
            if event.key == pygame.K_ESCAPE:
                done = True
            if event.key == pygame.K_LEFT:
                if len(game.figure) > 0 and game.figure[0].y > 15 and game.figure[0].x == 0:
                    game.score += 1
                    game.figure.popleft()
                elif game.score - 1 > 0:
                    game.score -= 1
                    game.fuck = True
                          
    screen.fill(WHITE)

    for i in range(game.height):
        for j in range(21 + 1):
            pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j * 36/21, game.y + game.zoom * i, 1, game.zoom], 1)
    for j in range(3):
        pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 1.4, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 3.25, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 6.55, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 8.33, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 10.11, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    pygame.draw.line(screen, GRAY, [game.x + game.zoom * 0, game.y + game.zoom* 0], [game.x + game.zoom* WIDTH, game.y + game.zoom * 0] )
    pygame.draw.line(screen, GRAY, [game.x + game.zoom * 0, game.y + game.zoom * 20], [game.x + game.zoom* WIDTH, game.y + game.zoom * 20] )
    pygame.draw.line(screen, RED, [game.x + game.zoom * 0, game.y + game.zoom * 15], [game.x + game.zoom* WIDTH, game.y + game.zoom * 15] )

    

    if game.figure is not None:
        for k in range(len(game.figure)):
            figure = game.figure[k]
            if figure.y < -2 or figure.y > 21:
                continue
            for p in range(len(figure.image)):
                if game.figure[k].image[p]:
                    pygame.draw.rect(screen, (0, 100 * key_width[p], 0), #screen and color
                        [real_pos[p] + game.zoom * (0.5/key_width[p] - 0.5), #x value of rectangle
                        game.y + game.zoom * (game.figure[k].y - 1) + 1,                                     #y value of rectangle
                        1.7* game.zoom * key_width[p] - 3, #Width of rectangle
                        max(game.zoom, game.zoom - 3 * game.figure[k].tune[p]) - 3]) #Height of rectangle
                if game.figure[k-1].noteName[p] != game.figure[k].noteName[p]:
                    screen.blit(font.render(game.figure[k].noteName[p], True, RED), [real_pos[p] + (game.zoom *0.5), 
                        game.y + game.zoom * (game.figure[k].y - 1)])
    if game.beat_bars is not None:
        for k in range(len(game.beat_bars)):
            pygame.draw.line(screen, GRAY, [game.x + game.zoom * 0, game.y + game.zoom * game.beat_bars[k] - 3], [game.x + game.zoom* WIDTH, game.y + game.zoom * game.beat_bars[k] - 3] )

    text = font1.render("Score: " + str(game.score), True, BLACK)
    text1 = font1.render("BPM: " + str(bpm), True, BLACK)
    text2 = font1.render("Time: " + str(counter//fps), True, BLACK)

    screen.blit(text, [0, 0])
    screen.blit(text1, [game.zoom * 15 ,0])
    screen.blit(text2, [game.zoom * 30 ,0])
    #text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))
    

    pygame.display.flip()
    clock.tick(fps)


pygame.quit()
