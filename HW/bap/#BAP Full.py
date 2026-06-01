import sys
import random
import pygame
import torch
import json
import pandas
import numpy as np
import struct
import pyaudio
import sounddevice

import Buttons_v2
import bapv1
import Piano_display    

def main():
    pygame.init()
    screen = pygame.display.set_mode((Piano_display.WIDTH*20, Piano_display.HEIGHT*20))
    pygame.display.set_caption("BAP")
    clock = pygame.time.Clock()
    #music = Piano_display.Music(Piano_display.HEIGHT, Piano_display.WIDTH, Piano_display.sheet)
    piano = Piano_display.Piano_display(Piano_display.WIDTH, Piano_display.HEIGHT, Piano_display.sheet)
    buttons = Buttons_v2.Buttons(Piano_display.WIDTH, Piano_display.HEIGHT)
    detector = bapv1.PureArrayDetector()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        

        screen.fill((255, 255, 255))
        piano.draw(screen)
        buttons.draw(screen)
        pygame.display.flip()
        clock.tick(Piano_display.bpm)
        detector.callback(sounddevice.rec(int(0.1 * bapv1.SAMPLE_RATE), samplerate=bapv1.SAMPLE_RATE, channels=1), 0, None, None)




while main() == True:
    main()