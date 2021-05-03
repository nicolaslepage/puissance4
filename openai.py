# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:38:28 2021

@author: Windows
"""

import gym
from gym import spaces
import numpy as np
from jeu import *

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


from rl.agents import DQNAgent
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy

from rl.memory import SequentialMemory

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self,depth):
      super(CustomEnv, self).__init__()
      # Define action and observation space
      # They must be gym.spaces objects
      
      # Example when using discrete actions:
      self.action_space = spaces.Discrete(7)
      # Example for using image as input:
      self.observation_space = spaces.Box(low=0, high=2, shape=(6,7), dtype=np.int32)
      
      self.state = np.zeros((6,7))
      
      self.depth = depth
      
      self.score = None
      
    
    def step(self, action):
        
        done = False
        
        # reinforcement learning agent plays
        
        ghostboard = np.copy(self.state)
        ghostboard.astype(np.int64)
        ghostboard = ai_moves(ghostboard,action,0)
        
        if ghostboard[0,0] == -1 :
            
            reward = -100
            
        else :
            
            self.state = ai_moves(self.state,action,0)
            
            winner = check_win(self.state)
            
            if winner != 0 :
                
                done = True
                
                if winner == 1 :
                    reward = 100
                    self.score = "reinf wins"
                
                elif winner == 2 :
                    reward = -10
                    self.score = "minimax wins"
                
                elif winner == 3 :
                    reward = 10
                    self.score = "draw"
            
            if not done :
                
            # minimax plays
            
                ghostboard2 = np.copy(self.state)
                ghostboard2.astype(np.int64)
                
                col_to_play, score = minimax(ghostboard2,self.depth,-1000000000,1000000000,True)
                
                self.state = ai_moves(self.state,col_to_play,1)
                              
                winner = check_win(self.state)
            
                if winner != 0 :
                    
                    done = True
                    
                    if winner == 1 :
                        reward = 100
                        self.score = "reinf wins"
                    
                    elif winner == 2 :
                        reward = -10
                        self.score = "minimax wins"
                    
                    elif winner == 3 :
                        reward = 10
                        self.score = "draw"
                                                
                else :
                    reward = evaluate(self.state, 0)
            
        info={}
        
        return self.state, reward, done, info
      
    def reset(self):
        # Reset the state of the environment to an initial state
        
        self.state = np.zeros((6,7))
        
        self.score = None
        
        return self.state
            
    def render(self):
        
        print(self.score)
        
        return self.score
    
    
class randomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self,depth):
      super(randomEnv, self).__init__()
      # Define action and observation space
      # They must be gym.spaces objects
      
      # Example when using discrete actions:
      self.action_space = spaces.Discrete(7)
      # Example for using image as input:
      self.observation_space = spaces.Box(low=0, high=2, shape=(6,7), dtype=np.int32)
      
      self.state = np.zeros((6,7))
      
      self.depth = depth
      
      self.score = None
      
    
    def step(self, action):
        
        done = False
        
        # reinforcement learning agent plays
        
        ghostboard = np.copy(self.state)
        ghostboard.astype(np.int64)
        ghostboard = ai_moves(ghostboard,action,0)
        
        if ghostboard[0,0] == -1 :
            
            reward = -100
                        
        else :
            
            self.state = ai_moves(self.state,action,0)
            
            winner = check_win(self.state)
            
            if winner != 0 :
                
                done = True
                
                if winner == 1 :
                    reward = 100
                    self.score = "reinf wins"
                
                elif winner == 2 :
                    reward = -10
                    self.score = "minimax wins"
                
                elif winner == 3 :
                    reward = 10
                    self.score = "draw"
            
            if not done :
                
            # minimax plays
            
                check = True
                z=0
                ghostboard2=np.copy(self.state)
                ghostboard2.astype(np.int64)
                while check :
                    if z > 100 :
                        break
                    rand_num= rd.randint(0,6)
                    check = ai_moves(ghostboard2,rand_num,1)[0,0] == -1
                    z+=1
                
                self.state = ai_moves(self.state,rand_num,1)
                              
                winner = check_win(self.state)
            
                if winner != 0 :
                    
                    done = True
                    
                    if winner == 1 :
                        reward = 100
                        self.score = "reinf wins"
                    
                    elif winner == 2 :
                        reward = -10
                        self.score = "minimax wins"
                    
                    elif winner == 3 :
                        reward = 10
                        self.score = "draw"
                        
                else :
                    reward = evaluate(self.state, 0)
            
        info={}
        
        return self.state, reward, done, info
      
    def reset(self):
        # Reset the state of the environment to an initial state
        
        self.state = np.zeros((6,7))
        
        self.score = None
              
        return self.state
        
    def render(self):
        # Render the environment to the screen
        print(self.score)
        
        return self.score
    
# env =  randomEnv(4)   


# states = np.shape(env.observation_space.sample())
# actions = env.action_space.n

def build_model(states, actions, layers, nodes):
    
    model = Sequential()
    
    model.add(Flatten(input_shape=(1,6,7)))
    
    for i in range (layers):
        model.add(Dense(nodes-20*i, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    
    return model


# model = build_model(states, actions, 5, 20)

def build_qagent(model, actions,steps):
    
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=0.0, nb_steps=steps)
    memory = SequentialMemory(limit=10000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=1000, target_model_update=1000)
    return dqn

def build_sagent(model, actions):
    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, policy=policy, nb_actions=actions)
    return sarsa


    