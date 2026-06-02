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
bpm = 140
Testing_variable_for_testing = (fps*fps//bpm//4)
speed = 1/Testing_variable_for_testing
detector = bapv1.PureArrayDetector()

hor_pos = [0,0.5,1,1.5,2,3,3.5,4,4.5,5,5.5,6,7,7.5,8,8.5,9,10,10.5,11,11.5,12,12.5,13,14,14.5,15,15.5,16,17,17.5,18,18.5,19,19.5,20,-10]
key_width = [1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1]
#buffer_sheet = [[0,16],]
                

sheet = [[0,0]]
testing_sheet = [
                 [1,4],
                 [0,4],
                 [2,5],
                 [3,2],
                 [4,3],
                 [5,7],
                 [6,10],
                 [7,9],
                 [8,3],
                 [9,4],
                 [10,4],
                 [11,4],
                 [12,4],
                 [13,4],
                 [14,4],
                 [15,4],
                 [16,4],
                 [17,4],
                 [18,4],
                 [19,4],
                 [20,4],
                 [21,4],
                 [22,4],
                 [23,4],
                 [24,4],
                 [25,4],
                 [26,4],
                 [27,4],
                 [28,4],
                 [29,4],
                 [30,4],
                 [31,4],
                 [32,4],
                 [33,4],
                 [34,4],
                 [35,4],
                 [36,4],] #note, length

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
        36,0,1,2,3,4,5,6,7,8,9,10,11,12,
         13,14,15,16,17,18,19,20,21,22,23,24,
         25,26,27,28,29,30,31,32,33,34,35
        ]
Note_list = [
                  "NULL","C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    ]
x = 0 
'''def check_note(game, inputty, x):
    setty = set()
    if len(game.figure) != 0:
        for i in game.figure[0].image:
            if i:
                setty.add(i)
            if game.figure[0].y > 15 and inputty.issubset(setty):
                game.figure.popleft()
                return True
            #if game.figure[0].y > 15:

    return None'''

class Figure:

    def __init__(self, Note):
        self.x = 0 #note
        self.y = 0
        self.tune = Note #Name of note
        self._len = Note[1]

        self._image = figures[Note[0]]

        self._note_names = Note_list[Note[0]]

    @property
    def image(self):
        return self._image

    @property
    def noteName(self):
        return self._note_names
    
    @property
    def length(self):
        return self._len

class Music:
    def __init__(self, height, width, sheet):
        self.score = 0
        self.x = 40
        self.y = 40
        self.note = 0
        self.delay = 16
        self.zoom = 33.333333
        self.figure = deque()
        self.fuck = False
        self.beat_bars = [0]

        self.sheet = sheet
        self.height = height
        self.width = width

    def new_figure(self):  #wait, append, track length
        print(self.delay)
        if self.note < len(self.sheet) and self.delay == 0:
            self.figure.append(Figure(self.sheet[self.note]))
            self.delay = Figure(self.sheet[self.note]).length
            self.note +=1
        self.delay -= 1
        
            
    
    def new_beatbar(self):
        self.beat_bars.append(0)

        if len(self.beat_bars) > 3:
            self.beat_bars.pop(0)
        
    def go_down(self):
        for k in range(len(self.figure)):
            self.figure[k].y += speed
        for h in range(len(self.beat_bars)):
            self.beat_bars[h] += speed
            
            
#get the chords and generate notes
chord_prog_2 = []
for chord in Buttons_v2.main():
    chord_prog_2.append((chord,4))
notes = generate.generate_music(generate.LSTMmodel, generate.chord_to_id, chord_prog_2*4, temperature=0.8)

sheet = make_sheet(notes,36)
total_sheet = testing_sheet

real_pos = [40 + 33.333 * hor_pos[p] * 36/21 + 2 for p in range(37)]

# Initialize the game engine
pygame.init()
print(" starting ")

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)

size = (1280, 720) # width, height
flags = pygame.FULLSCREEN
screen = pygame.display.set_mode(size, flags, vsync=1)
font = pygame.font.SysFont('Calibri', 40, True, False)
font1 = pygame.font.SysFont('Calibri', 50, True, False)
pygame.display.set_caption("Jazz")

# Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()

game = Music(HEIGHT, WIDTH, total_sheet) #height, width, sheet
counter = 0
tally = 0
pressing_down = False

background = pygame.Surface(size)
background.fill(WHITE)

for i in range(game.height):
    for j in range(21 + 1):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 36/21, game.y + game.zoom * i, 1, game.zoom], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 1.4, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 3.25, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 6.55, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 8.33, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 10.11, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    pygame.draw.line(background, GRAY, [game.x + game.zoom * 0, game.y + game.zoom* 0], [game.x + game.zoom* WIDTH, game.y + game.zoom * 0] )
    pygame.draw.line(background, GRAY, [game.x + game.zoom * 0, game.y + game.zoom * 20], [game.x + game.zoom* WIDTH, game.y + game.zoom * 20] )
    pygame.draw.line(background, RED, [game.x + game.zoom * 0, game.y + game.zoom * 15], [game.x + game.zoom* WIDTH, game.y + game.zoom * 15] )

sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                blocksize=FFT_SIZE, callback=detector.callback).start()


while not done:
    counter += 1
    if counter > 100000:
        counter = 0

    if counter % (Testing_variable_for_testing) == 0 or pressing_down:
        game.new_figure()
        if len(game.figure) != 0 and game.figure[0].y > 19:
            game.figure.popleft()
        
        #if not game.fuck and (game.score - 1) >= 0:
            #game.score -= 1
        game.fuck = False
    
    if counter % (Testing_variable_for_testing*Testing_variable_for_testing) == 0:
        game.new_beatbar()
            

    game.go_down()
    '''if check_note(game, inputty, game.score) and inputty and counter > 16 :
        game.score += 1'''
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

    screen.blit(background, (0,0))
    
    if game.figure is not None:
        for figure in game.figure:
            if figure.y < -2 or figure.y > 21:
                continue
            if figure.length != 0:
                pygame.draw.rect(screen, (0, 100 * key_width[figure.image], 0), #screen and color
                    [real_pos[figure.image] + game.zoom * (0.5/key_width[figure.image] - 0.5), #x value of rectangle
                    game.y + game.zoom * (figure.y - figure.length) + 1, #y value of rectangle
                    1.7* game.zoom * key_width[figure.image] - 3, #Width of rectangle
                    game.zoom * figure.length - 3]) #Height of rectangle
                screen.blit(font.render(figure.noteName, True, RED), [real_pos[figure.image] + (game.zoom *0.5), 
                    game.y + game.zoom * (figure.y - 1)])
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
