# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:04:10 2021

@author: Windows
"""

###### reinforcement leanrning env #######

from reinf_agent import *
from jeu import *
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import time


def iteration(agent,depth,i):
    running = True 
    single_jeu= jeu()
    
    firstaction = agent.agent_start(single_jeu.board.flatten())

    
    single_jeu.move(firstaction,0)
    reward_sum = 0
    
    while running :
        
             
        
        
        # print(single_jeu.board)
        # time.sleep(1)
        
        # minimax agent move
        
        single_jeu.ai(depth)
        
                
        # check = True
        # z=0
        # while check :
        #     if z > 100 :
        #         break
        #     rand_num= rd.randint(0,6)
        #     ghostboard=np.copy(single_jeu.board)
        #     ghostboard.astype(np.int64)
        #     check = ai_moves(ghostboard,rand_num,1)[0,0] == -1
            
        # single_jeu.move(rand_num,1)
        
        # if i%1000 == 0 :
        #     print(single_jeu.board)
        
        #check minimax win
        
        winner=int(single_jeu.check())

        running = winner == 0
        
        if winner == 1 :
            agent.agent_end(1000)
        if winner == 2 :
            agent.agent_end(-10)
        if winner == 3:
            agent.agent_end(0)
            
        if not running :
            break
        
        # reinf agent move
        
        reward = evaluate(single_jeu.board, 0)
        reward_sum += reward
        ghostboard=np.copy(single_jeu.board)
        ghostboard.astype(np.int64)
        col = agent.agent_step4(reward,ghostboard.flatten(),ghostboard)

        single_jeu.move(col,0)
        
        #check minimax win
        
        winner=int(single_jeu.check())

        running = winner == 0
        
        if winner == 1 :
            agent.agent_end(1000)
        if winner == 2 :
            agent.agent_end(-10)
        if winner == 3:
            agent.agent_end(0)
    
    return winner, reward_sum
    

        
        
        
        
    pass

def train(episode,units, alpha, betam, betav, epsilon, gama, tau):
    agent_info = {
             'network_config': {
                 'state_dim': 42,
                 'num_hidden_units': units,
                 'num_hidden_layers': 3,
                 'num_actions': 7
             },
             'optimizer_config': {
                 'step_size': alpha, 
                 'beta_m': betam, 
                 'beta_v': betav,
                 'epsilon': epsilon
             },
             'replay_buffer_size': 32,
             'minibatch_sz': 32,
             'num_replay_updates_per_step': 4,
             'gamma': gama,
             'tau': tau,
             'seed': 0}

    # Initialize agent
    agent = Agent()
    agent.agent_init(agent_info)
    depth = 4
    rewards= 0
    sum_wins = 0
    
    for i in range (episode) :
        
        winner,reward=iteration(agent,depth,i)
        rewards += reward
        
        # print("winner is = " + str(winner))
                
        if winner == 1 :
            sum_wins += 1
            
        # if i % 1000 == 0 :
        #     print("ep number = " +str(i))
        
        # if i % 10000 == 0 :
                       
        #     print("wins in the last 1000 games = " + str(sum_wins))
        #     print("average reward of last 1000 games = " + str(rewards))
        #     rewards= 0
        #     sum_wins = 0
    
    print("tatol reward = " +str(rewards) + "                                  number of games won in 1000 episodes = " + str(sum_wins))

for units in range (1,9):
    for alpha in range (0,6):
        for epsilon in range (0,5):
            for tau in range (1,5):
                print("training with following parameters =")
                print( "units = " + str(2**units) + "     learning rate = " + str(10**(-alpha)) + "     epsi = " + str(10**(-epsilon)) + "      tau = " + str(10**(tau-5)))
                print("")
                for i in range (5):
                    train(1000, 2**units , 10**(-alpha-3), 0.9, 0.999, 10**(-epsilon-5), 0.99, 10**(-tau))
                print("")             
                



    
    



    
