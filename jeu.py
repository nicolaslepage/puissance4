# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 19:54:49 2021

@author: npfle
"""

import numpy as np
import random as rd
import math as mt
from numba import jit    # import the types


######## jitted functions, not in jeu class ###########

@jit(nopython=True)
def check_win(board):
    c = 0
    
    for i in range (0,6):
        for j in range(0,4):
            if (board[i,j] == 1 or board[i,j] == 2) and (board[i,j] == board[i,j+1] == board[i,j+2] == board[i,j+3]):
                return int(board[i,j])
              
    for j in range (0,7):
        for i in range(0,3):
            if (board[i,j] == 1 or board[i,j] == 2) and (board[i,j] == board[i+1,j] == board[i+2,j] == board[i+3,j]):
                return int(board[i,j])
            
    for i in range (0,3):
        for j in range (0,4):
            if (board[i,j] == 1 or board[i,j] == 2) and (board[i,j] == board[i+1,j+1] == board[i+2,j+2] == board[i+3,j+3]):
                return int(board[i,j])
    
    for i in range (3,6):
        for j in range (0,4):
            if (board[i,j] == 1 or board[i,j] == 2) and (board[i,j] == board[i-1,j+1] == board[i-2,j+2] == board[i-3,j+3]):
                return int(board[i,j])
            
    for i in range (0,6):
        for j in range (0,7):
            if board[i,j] == 0 :
                c+=1
    if c==0 :
        return int(3)
    else :
        return int(0)


@jit(nopython=True)
def evaluate(board,turn) :        
    
    score = 0
    antiturn = (turn + 1) % 2


    score += horizontal_check(board,turn)
    score -= horizontal_check(board,antiturn)
    
    score += vertical_check(board,turn)
    score -= vertical_check(board,antiturn)
    
    score += diag1(board,turn)
    score -= diag1(board,antiturn)
    
    score += diag2(board,turn)
    score -= diag2(board,antiturn)  
    
    return score

@jit(nopython=True)
def evaluate_reinf(board,turn) :        
    
    score = 0
    antiturn = (turn + 1) % 2


    score += (horizontal_check(board,turn)*2)**2
    score -= horizontal_check(board,antiturn)
    
    score += (vertical_check(board,turn)*2)**2
    score -= vertical_check(board,antiturn)
    
    score += (diag1(board,turn)*2)**2
    score -= diag1(board,antiturn)
    
    score += (diag2(board,turn)*2)**2
    score -= diag2(board,antiturn)  
    
    return score

@jit(nopython=True)
def horizontal_check(board,turn):
    score=0
    
    for i in range (0,6):
        j=0
        while j <= 5 :
            sub_score=0
            while board[i,j] == turn+1 and board[i,j]==board[i,j+1] :
                sub_score+=1
                j+=1
                if j>=6 or sub_score > 2 :
                    break
            if sub_score == 0:
                j+=1
            score+=sub_score**2
    
    return score

@jit(nopython=True)
def vertical_check(board,turn):
    
    score = 0
    
    for j in range (0,7):
        i=0
        while i <= 4 :
            sub_score=0
            while board[i,j] == turn+1 and board[i,j]==board[i+1,j] :
                sub_score+=1
                i+=1
                if i>=5 or sub_score > 2:
                    break
            if sub_score == 0:
                i+=1
            score+=sub_score**2
            
    return score

@jit(nopython=True)
def diag1(board,turn):
    
    score = 0
    
    istart=0
    jstart=0
    flag1=True
    flag2=True
    i=istart
    j=jstart
    while flag1:
        
        sub_score=0
        while board[i,j] == turn+1 and board[i,j]==board[i+1,j+1] :
            sub_score+=1
            i+=1
            j+=1
            
            if i>=5 or j>=6 or sub_score > 2:
                break
        
        if sub_score == 0:
            i+=1
            j+=1  
        
        if i>=5 or j>=6:
            
            if i==5 and j == 4:
                flag1= False
                        
            if jstart<3 and flag2:
                jstart+=1
                i=istart
                j=jstart
            else :
                flag2 = False
                jstart=0
                istart+=1
                i=istart
                j=jstart
        
        score+=sub_score**2
    
    return score

@jit(nopython=True)
def diag2(board,turn):
    
    score=0
    
    istart=5
    jstart=0
    flag1=True
    flag2=True
    i=istart
    j=jstart
    
    while flag1:
        
        sub_score=0
        while board[i,j] == turn+1 and board[i,j]==board[i-1,j+1] :
            sub_score+=1
            i-=1
            j+=1
            
            if i<=0 or j>=6 or sub_score > 2:
                break
        
        if sub_score == 0:
            i-=1
            j+=1  
        
        if i<=0 or j>=6:
            
            if i==0 and j == 4:
                flag1= False
                        
            if jstart<3 and flag2:
                jstart+=1
                i=istart
                j=jstart
            else :
                flag2 = False
                jstart=0
                istart-=1
                i=istart
                j=jstart
                
        score+=sub_score**2
    
    return score

@jit(nopython=True)
def generate_child(board,turn):
    children = np.zeros((7,6,7),dtype=np.int64)
    for i in range (7):
        ghostboard=np.copy(board)
        ghostboard.astype(np.int64)
        children[i,:,:] = ai_moves(ghostboard,i,turn)
    return children
    
@jit(nopython=True)
def ai_moves(board,col,turn):
    for r in range (0,6) :
        if board[5-r,col] == 0:
            board[5-r,col] = turn+1
            return board.astype(np.int64)
    return np.zeros((6,7),dtype=np.int64)-np.ones((6,7),dtype=np.int64)

@jit(nopython=True)       
def minimax(board,depth,alpha,beta,maximising_player):
    
    if maximising_player :
        turn = 1
    else :
        turn = 0

    win_indicator = check_win(board)
    
    if depth == 0 or win_indicator != 0 :
        if win_indicator == 1 :
            return None, -10000
        elif win_indicator == 2 :
            return None, 10000
        elif win_indicator == 3 :
            return None, 0
        else:
            return None, evaluate(board,1)
        
    children = generate_child(board,turn)
                
    if maximising_player :
        
        maxeval = -mt.inf
        
        column=0
        
        for i in range (7):
            
            child = children[i,:,:]
            
            if child[0,0]>=0:
                
                evalu=minimax(child,depth-1,alpha,beta,False)[1]
                
                if evalu > maxeval :
                    
                    column = i
                    maxeval = evalu
                    
                alpha=max(alpha,evalu)
                if beta <= alpha:
                    break
        return column, maxeval
    
    else :
        
        mineval = mt.inf
        
        column=0
        
        for i in range (7):
            
            child = children[i,:,:]
            
            if child[0,0]>=0:
                
                evalu=minimax(child,depth-1,alpha,beta,True)[1]
                
                if evalu < mineval :
                    
                    mineval = evalu
                    column = i
                    
                beta=min(beta,evalu)
                
                if beta <= alpha:
                    break
        return column, mineval

#####class and methods 
            
class jeu :
    
    def __init__(self):
        self.board = np.zeros((6,7),dtype=np.int64)
        
    def move(self,col,turn):
        for r in range (0,6) :
            if self.board[5-r,col] == 0:
                self.board[5-r,col] = turn+1
                return True
        return False
    
    def ai(self,depth):
        ghostboard = np.copy(self.board)
        ghostboard.astype(np.int64)
        
        col_to_play, score = minimax(ghostboard,depth,-1000000000,1000000000,True)
        
        if col_to_play != None :
            self.move(col_to_play,1)
            
    def check_score(self,turn):
        return evaluate(self.board,turn)
        
    def check(self):
        return check_win(self.board)
     
        
        
####### tests ###########
        
#jeu1 = jeu()
#jeu1.move(2,0)
#jeu1.ai(6)
#print(jeu1.board)