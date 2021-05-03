# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:52:29 2021

@author: npfle
"""

import pygame as pg
import numpy as np
import random as rd
import time
from jeu import *


# init du jeu

jeu1 = jeu()
            
pg.init()

size_square=200

xsize= 7*size_square
ysize= 6*size_square
screen = pg.display.set_mode((xsize,ysize))
font = pg.font.Font('freesansbold.ttf', 64)

running = True


def draw(board):
    screen.fill((50, 50, 150))
    for r in range(0,7):
        for c in range(0,6):
            pg.draw.circle(screen, (230,230,230), (int(r*size_square+size_square/2), int(c*size_square+size_square/2)), 80)
            
            if jeu1.board[c,r] == 1 :
                pg.draw.circle(screen, (230,50,50), (int(r*size_square+size_square/2), int(c*size_square+size_square/2)), 80)
            elif jeu1.board[c,r] == 2 :
                pg.draw.circle(screen, (200,200,0), (int(r*size_square+size_square/2), int(c*size_square+size_square/2)), 80)

winner = int(0)
mean_time= 0


while running:
    
    for event in pg.event.get():
        
        if event.type == pg.QUIT:
            running = False
            pg.display.quit()
            
        if event.type == pg.MOUSEBUTTONDOWN:
            posx = event.pos[0]
            row = int(np.floor(posx/size_square))
            check = jeu1.move(row,0)
            if check :
                draw(jeu1.board)
                pg.display.update()
                jeu1.ai(10)
            winner=int(jeu1.check())
    
    draw(jeu1.board)
    running = winner == 0
    
    if winner == 1 :
        screen.fill((230,50,50))
    if winner == 2 :
        screen.fill((200,200,0))
    if winner == 3:
        screen.fill((230, 150, 0))
        
        
    pg.display.update()
    
    pg.time.wait(1000)
    
    draw(jeu1.board)
    
    if not running :
        pg.time.wait(1000)
    
        draw(jeu1.board)
        pg.display.update()
    
        pg.time.wait(5000)
        pg.display.quit()
