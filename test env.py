# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:48:51 2021

@author: Windows
"""
from openai import *

import time

import random as rd

env =  CustomEnv(9)   

summ = 0


for i in range (100) :
    
    start =time.time()

    for i in range (100):
        action = rd.randint(0,6)
        done = env.step(action)[2]
        # # print(state)
        # # print("")
        # # print("reward was = " +str(reward))
        # # print("")
        # x,y = np.shape(env.state)
        # number1 = 0
        # number2 = 0
        # for i in range (x):
        #     for j in range (y):
        #         if state[i,j]==1:
        #             number1  += 1
        #         if state[i,j]==2:
        #             number2  += 1
        
        # # print("numbers of 1 = " + str(number1) + "        numbers of 2 = " + str(number2) )
        # # print("")
                
        if done :
            env.reset()
            
    end =time.time()
    summ+=end

    print("time per step =" + str((end-start)/100))     

print("total average =" + str((end-start)/10000))  