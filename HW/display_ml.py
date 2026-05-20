
import pygame
import random
import numpy as np
# import sys
# sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//ML')
# import generate


WIDTH = 36
HEIGHT = 20
bpm = 80 #bpm

colors = [
    (0, 0, 0),
]
hor_pos = [0,0.5,1,1.5,2,3,3.5,4,4.5,5,5.5,6,7,7.5,8,8.5,9,10,10.5,11,11.5,12,12.5,13,14,14.5,15,15.5,16,17,17.5,18,18.5,19,19.5,20]
key_width = [1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1]

# sheet = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#          [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#          [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#          [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],
#          [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],
#          [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
#          [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
#          [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],
#          [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
#          [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#          ]
col_start = 48
# notes = generate.generate_music(generate.LSTMmodel, generate.chord_to_id, generate.iiVI_long, temperature=0.95)
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

class Figure:

    figures = [
        [1,2,3,4,5,6,7,8,9,10,11,12,
         13,14,15,16,17,18,19,20,21,22,23,24,
         25,26,27,28,29,30,31,32,33,34,35,36]
        ]
    Note_list = [
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    ]

    def __init__(self, Notes):
        self.x = 0 #note
        self.y = 0
        self.length = 1 #lenght of note
        self.color = 0
        self.tune = Notes #Name of note


    def image(self):
        return np.array([self.figures]) * np.array([self.tune])
    
    def direction(self):
        #print([self.Note_list[i] for i in range(len(self.direc)) if self.direc[i] == 1])
        return [self.Note_list[i] if self.tune[i] != 0 else '' for i in range(len(self.tune))]


class Music:
    def __init__(self, height, width, sheet):
        self.state = "start"
        self.score = 0
        self.height = 0
        self.width = 0
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = []
        self.fuck = False
        self.note = 0
        #self.delay = 0
        self.beat_bars = [0]
    
        self.sheet = sheet
        self.height = height
        self.width = width

    def new_figure(self):
            if self.note < len(self.sheet): #and self.delay == 0:
                self.figure.append(Figure(self.sheet[self.note]))
                #self.delay = self.sheet[self.note][0] 
                self.note += 1
            #self.delay -= 1
                
            if self.figure[0].y > 19 + self.figure[0].length:             #test was game
                self.figure.pop(0)
                if not self.fuck and (self.score - 1) >= 0:
                    self.score -= 1
                self.fuck = False
    
    def new_beatbar(self):
        self.beat_bars.append(0)

        if len(self.beat_bars) > 5:
            self.beat_bars.pop(0)
        



    def go_down(self):
        for k in range(len(self.figure)):
            self.figure[k].y += (1/16)
        for h in range(len(self.beat_bars)):
            self.beat_bars[h] += (1/16)

def display_start(sheet):

    # Initialize the game engine
    pygame.init()
    print(" starting ")

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    RED = (255, 0, 0)

    size = (1100, 600) # width, height
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Jazz")

    # Loop until the user clicks the close button.
    done = False
    clock = pygame.time.Clock()

    game = Music(HEIGHT, WIDTH, sheet) #height, width
    counter = 0

    pressing_down = False

    while not done:
        counter += 1
        if counter > 100000:
            counter = 0

        if counter % (16) == 0 or pressing_down:
            if game.state == "start":
                game.new_figure()
        
        if counter % (16*16) == 0:
            game.new_beatbar()
                
        
        if counter % (1) == 0 or pressing_down:
            if game.state == "start":
                game.go_down()
                

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game.__init__(HEIGHT, WIDTH)
                if event.key == pygame.K_LEFT:
                    if game.figure[0].y > 15 and game.figure[0].x == 0:
                        game.score += 1
                        game.figure.pop(0)
                    elif game.score - 1 > 0:
                        game.score -= 1
                        game.fuck = True
                if event.key == pygame.K_UP:
                    if game.figure[0].y > 15 and game.figure[0].x == 1:
                        game.score += 1
                        game.figure.pop(0)
                    elif game.score - 1 > 0:
                        game.score -= 1
                        game.fuck = True
                if event.key == pygame.K_DOWN:
                    if game.figure[0].y > 15 and game.figure[0].x == 2:
                        game.score += 1
                        game.figure.pop(0)
                    elif game.score - 1 > 0:
                        game.score -= 1
                        game.fuck = True
                if event.key == pygame.K_RIGHT:
                    if game.figure[0].y > 15 and game.figure[0].x == 3:
                        game.score += 1
                        game.figure.pop(0)
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

        font = pygame.font.SysFont('Calibri', 10, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)

        if game.figure is not None:
            for i in range(1):
                for j in range(WIDTH):
                    p = i * WIDTH + j
                    for k in range(len(game.figure)):              
                        if p in abs(game.figure[k].image()) - 1:
                            pygame.draw.rect(screen, colors[game.figure[k].color],
                                [game.x + game.zoom * hor_pos[p] * 36/21 + 2 + game.zoom * (0.5/key_width[p] - 0.5),
                                game.y + game.zoom * (i + game.figure[k].y - game.figure[k].length) + 1,
                                1.7* game.zoom * key_width[p] - 3, max(game.zoom, game.zoom - 3 * game.figure[k].tune[p]) - 3])
        if game.figure is not None:
            for k in range(len(game.figure)):
                for h in range(len(game.figure[k].direction())):
                    if game.figure[k].direction()[h]:
                        screen.blit(font.render(game.figure[k].direction()[h], True, WHITE), [game.x + game.zoom * hor_pos[h] * 36/21 + 2 + (game.zoom *0.5),
                            game.y + game.zoom * (game.figure[k].y - game.figure[k].length)])

        if game.beat_bars is not None:
            for k in range(len(game.beat_bars)):
                pygame.draw.line(screen, GRAY, [game.x + game.zoom * 0, game.y + game.zoom * game.beat_bars[k] - 20], [game.x + game.zoom* WIDTH, game.y + game.zoom * game.beat_bars[k] - 20] )


        text = font1.render("Score: " + str(game.score), True, BLACK)
        text1 = font1.render("BPM: " + str(bpm), True, BLACK)


        screen.blit(text, [0, 0])
        screen.blit(text1, [game.zoom * 20 ,0])
        #text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

        pygame.display.flip()
        clock.tick(bpm)

    pygame.quit()
    return None