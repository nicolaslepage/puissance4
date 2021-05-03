# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:52:07 2021

@author: Windows
"""

from openai import *
import keras

steps = 5000000

env =  randomEnv(6)   


states = np.shape(env.observation_space.sample())
actions = env.action_space.n

model = build_model(states, actions, 10, 248)

dqn = build_qagent(model, actions, steps)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=steps/4 ,decay_rate=0.9)

dqn.compile(Adam(lr=1e-5), metrics=['mae'])
history=dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)

data = []

for i in range (len(history.history['episode_reward'])-100):
    temp_data = 0
    for j in range (100) :
        temp_data += history.history['episode_reward'][i+j]
    data.append(temp_data)





env2 = CustomEnv(3)

history=dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)


for i in range (len(history.history['episode_reward'])-100):
    temp_data = 0
    for j in range (100) :
        temp_data += history.history['episode_reward'][i+j]
    data.append(temp_data)





env3 = CustomEnv(4)
history=dqn.fit(env3, nb_steps=steps, visualize=False, verbose=1)

for i in range (len(history.history['episode_reward'])-100):
    temp_data = 0
    for j in range (100) :
        temp_data += history.history['episode_reward'][i+j]
    data.append(temp_data)





env4 = CustomEnv(5)

history=dqn.fit(env4, nb_steps=steps, visualize=False, verbose=1)


for i in range (len(history.history['episode_reward'])-100):
    temp_data = 0
    for j in range (100) :
        temp_data += history.history['episode_reward'][i+j]
    data.append(temp_data)
    
    
    
    

env5 = CustomEnv(6)

history=dqn.fit(env5, nb_steps=steps, visualize=False, verbose=1)

for i in range (len(history.history['episode_reward'])-100):
    temp_data = 0
    for j in range (100) :
        temp_data += history.history['episode_reward'][i+j]
    data.append(temp_data)
    
    
    
    

env6 = CustomEnv(7)

history=dqn.fit(env6, nb_steps=steps, visualize=False, verbose=1)

for i in range (len(history.history['episode_reward'])-100):
    temp_data = 0
    for j in range (100) :
        temp_data += history.history['episode_reward'][i+j]
    data.append(temp_data)

plt.plot(data)
plt.title('model accuracy')
plt.ylabel('reward per ep')
plt.xlabel('episode')
plt.legend(['train'], loc='upper left')
plt.show()

scores = dqn.test(env, nb_episodes=10, visualize=True)

scores = dqn.test(env3, nb_episodes=10, visualize=True)
scores = dqn.test(env3, nb_episodes=10, visualize=True)
scores = dqn.test(env4, nb_episodes=10, visualize=True)
scores = dqn.test(env5, nb_episodes=10, visualize=True)
scores = dqn.test(env6, nb_episodes=10, visualize=True)
    