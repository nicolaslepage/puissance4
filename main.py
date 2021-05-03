# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 19:52:41 2021

@author: npfle
"""

import pygame as pg
import numpy as np
import random as rd
import time

# init du jeu

jeu1 = jeu()

# type de jeu

#invalid_choice = True
#
#print("what type of game should you play ? 0-1-2")
#type_of_game = input()
#
#while invalid_choice :
#    
#    if type(type_of_game) != type(int(1)):
#        print("invalid choice, must be int")
#        print("what type of game should you play ? 0-1-2")
#        type_of_game = int(input())
#    
#    else :
#    
#        if type_of_game == 0 :
#            print("you have chosen IA vs IA")
#            invalid_choice = False
#            
#        elif type_of_game == 1 :
#            print("you have chosen 1 vs IA")
#            invalid_choice = False
#            
#        elif type_of_game == 2 :
#            print("you have chosen 1 vs 1")
#            invalid_choice = False
#            
#        else :
#            print("invalid choice")
#            print("what type of game should you play ? 0-1-2")
#            type_of_game = int(input())
            
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
                pg.draw.circle(screen, (230,230,50), (int(r*size_square+size_square/2), int(c*size_square+size_square/2)), 80)

turn = int(0)
winner = int(0)
mean_time= 0

while running:
    
    start_time = time.time()
    
    for event in pg.event.get():
        
        if event.type == pg.QUIT:
            running = False
            pg.display.quit()
            
        if event.type == pg.MOUSEBUTTONDOWN:
            posx = event.pos[0]
            row = int(np.floor(posx/size_square))
            check = jeu1.move(row,turn)
            print("score for player = " +str(turn))
            print(jeu1.check_score(turn))
            if check :
                turn+=1
            winner=int(jeu1.check())
                        
    draw(jeu1.board)
    
    running = winner == 0
    
    if winner == 1 :
        screen.fill((230,50,50))
    if winner == 2 :
        screen.fill((230, 230, 50))
    if winner == 3:
        screen.fill((230, 150, 0))
        
    pg.display.update()
    
    if not running :
        pg.time.wait(1000)
        pg.display.quit()
    
    turn = int(turn % 2)
    
    if mean_time == 0 :
        mean_time = time.time() - start_time
    elif running:
        mean_time = (mean_time + (time.time() - start_time))/2

print (mean_time)

            
    
        
    
    
