# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:39:07 2021

@author: npfle
"""

# Do not modify this cell!

# Import necessary libraries
# DO NOT IMPORT OTHER LIBRARIES - This will break the autograder.
import numpy as np
import matplotlib.pyplot as plt

from rl_glue import RLGlue
from environment import BaseEnvironment

from lunar_lander import LunarLanderEnvironment

from agent import BaseAgent

from collections import deque

from copy import deepcopy

from tqdm import tqdm

import os 

import shutil

from plot_script import plot_result


#############        defining neural network class             ##################

class ActionValueNetwork:
    # Work Required: Yes. Fill in the layer_sizes member variable (~1 Line).
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        
        self.rand_generator = np.random.RandomState(network_config.get("seed"))
        
        # Specify self.layer_size which shows the number of nodes in each layer
        ### START CODE HERE (~1 Line)
        self.layer_sizes = None
        ### END CODE HERE
        
        self.layer_sizes =np.zeros((3), dtype = int)
        
        for i in range (3) :
            self.layer_sizes[i] = self.num_hidden_units
        
        self.layer_sizes[0] = self.state_dim
        self.layer_sizes[-1] = self.num_actions
        
        
        # Initialize the weights of the neural network
        # self.weights is an array of dictionaries with each dictionary corresponding to 
        # the weights from one layer to the next. Each dictionary includes W and b
        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights[i]['W'] = self.init_saxe(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))
    
    # Work Required: No.
    def get_action_values(self, s):
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """
        
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        q_vals = np.dot(x, W1) + b1

        return q_vals
    
    # Work Required: No.
    def get_TD_update(self, s, delta_mat):
        """
        Args:
            s (Numpy array): The state.
            delta_mat (Numpy array): A 2D array of shape (batch_size, num_actions). Each row of delta_mat  
            correspond to one state in the batch. Each row has only one non-zero element 
            which is the TD-error corresponding to the action taken.
        Returns:
            The TD update (Array of dictionaries with gradient times TD errors) for the network's weights
        """

        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)

        # td_update has the same structure as self.weights, that is an array of dictionaries.
        # td_update[0]["W"], td_update[0]["b"], td_update[1]["W"], and td_update[1]["b"] have the same shape as 
        # self.weights[0]["W"], self.weights[0]["b"], self.weights[1]["W"], and self.weights[1]["b"] respectively
        td_update = [dict() for i in range(len(self.weights))]
         
        v = delta_mat
        td_update[1]['W'] = np.dot(x.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
        
        v = np.dot(v, W1.T) * dx
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
                
        return td_update
    
    # Work Required: No. You may wish to read the relevant paper for more information on this weight initialization
    # (Exact solutions to the nonlinear dynamics of learning in deep linear neural networks by Saxe, A et al., 2013)
    def init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor
    
    # Work Required: No.
    def get_weights(self):
        """
        Returns: 
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)
    
    # Work Required: No.
    def set_weights(self, weights):
        """
        Args: 
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)
        
#############        end of definition of neural network class             ##################

#############        calling neural net class                               ##################

network_config = {
    "state_dim": 5,
    "num_hidden_units": 20,
    "num_actions": 3
}



test_network = ActionValueNetwork(network_config)

#############        calling neural net class                               ##################

#############        defining class for backprob                               ##################

class Adam():
    # Work Required: Yes. Fill in the initialization for self.m and self.v (~4 Lines).
    def __init__(self, layer_sizes, optimizer_info):
        
        self.layer_sizes = layer_sizes

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")
        
        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(1, len(self.layer_sizes))]
        self.v = [dict() for i in range(1, len(self.layer_sizes))]
        
        for i in range(0, len(self.layer_sizes) - 1):
            ### START CODE HERE (~4 Lines)
            # Hint: The initialization for m and v should look very much like the initializations of the weights
            # except for the fact that initialization here is to zeroes (see description above.)
            self.m[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.m[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.v[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            ### END CODE HERE
            
        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to 
        # the time step t. We can calculate these powers using an incremental product. At initialization then, 
        # beta_m_product and beta_v_product should be ...? (Note that timesteps start at 1 and if we were to 
        # start from 0, the denominator would be 0.)
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v
    
    # Work Required: Yes. Fill in the weight updates (~5-7 lines).
    def update_weights(self, weights, td_errors_times_gradients):
        """
        Args:
            weights (Array of dictionaries): The weights of the neural network.
            td_errors_times_gradients (Array of dictionaries): The gradient of the 
            action-values with respect to the network's weights times the TD-error
        Returns:
            The updated weights (Array of dictionaries).
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                ### START CODE HERE (~5-7 Lines)
                # Hint: Follow the equations above. First, you should update m and v and then compute 
                # m_hat and v_hat. Finally, compute how much the weights should be incremented by.
                # self.m[i][param] = None
                # self.v[i][param] = None
                
                self.m[i][param] = self.beta_m*self.m[i][param] + (1-self.beta_m)*td_errors_times_gradients[i][param]
                self.v[i][param] = self.beta_v*self.v[i][param] + (1-self.beta_v)*td_errors_times_gradients[i][param]**2
                
                # m_hat = None
                # v_hat = None
                
                m_hat = self.m[i][param]/(1-self.beta_m_product )
                v_hat = self.v[i][param]/(1-self.beta_v_product )
                
                weight_update = (self.step_size/(np.sqrt(v_hat)+self.epsilon))*m_hat
                ### END CODE HERE
                
                weights[i][param] = weights[i][param] + weight_update
        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to 
        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v
        
        return weights
    

#############       end of  definition of neural net class class for backprob                               ##################

#############       calling backprob backprob                               ##################

network_config = {"state_dim": 5,
                  "num_hidden_units": 2,
                  "num_actions": 3
                 }

optimizer_info = {"step_size": 0.1,
                  "beta_m": 0.99,
                  "beta_v": 0.999,
                  "epsilon": 0.0001
                 }

network = ActionValueNetwork(network_config)
test_adam = Adam(network.layer_sizes, optimizer_info)


## Test Code for Adam update_weights() ##

network_config = {"state_dim": 5,
                  "num_hidden_units": 2,
                  "num_actions": 3
                 }

optimizer_info = {"step_size": 0.1,
                  "beta_m": 0.99,
                  "beta_v": 0.999,
                  "epsilon": 0.0001
                 }

network = ActionValueNetwork(network_config)
test_adam = Adam(network.layer_sizes, optimizer_info)


rand_generator = np.random.RandomState(0)

# Initialize m and v
test_adam.m[0]["W"] = rand_generator.normal(0, 1, (5, 2))
test_adam.m[0]["b"] = rand_generator.normal(0, 1, (1, 2))
test_adam.m[1]["W"] = rand_generator.normal(0, 1, (2, 3))
test_adam.m[1]["b"] = rand_generator.normal(0, 1, (1, 3))

test_adam.v[0]["W"] = np.abs(rand_generator.normal(0, 1, (5, 2)))
test_adam.v[0]["b"] = np.abs(rand_generator.normal(0, 1, (1, 2)))
test_adam.v[1]["W"] = np.abs(rand_generator.normal(0, 1, (2, 3)))
test_adam.v[1]["b"] = np.abs(rand_generator.normal(0, 1, (1, 3)))

# Specify weights
weights = [dict() for i in range(1, len(test_adam.layer_sizes))]
weights[0]["W"] = rand_generator.normal(0, 1, (5, 2))
weights[0]["b"] = rand_generator.normal(0, 1, (1, 2))
weights[1]["W"] = rand_generator.normal(0, 1, (2, 3))
weights[1]["b"] = rand_generator.normal(0, 1, (1, 3))

# Specify g
g = [dict() for i in range(1, len(test_adam.layer_sizes))]
g[0]["W"] = rand_generator.normal(0, 1, (5, 2))
g[0]["b"] = rand_generator.normal(0, 1, (1, 2))
g[1]["W"] = rand_generator.normal(0, 1, (2, 3))
g[1]["b"] = rand_generator.normal(0, 1, (1, 3))

# Update weights
updated_weights = test_adam.update_weights(weights, g)


#############       end of calling backprob backprob                               ##################


#############      class to virtaully replay moves and train more effectively                      ##################


class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.              
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator. 
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):
        """
        Args:
            state (Numpy array): The state.              
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.           
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)
    
#############   end of class to virtaully replay moves and train more effectively                      ##################

#############      softmax function                     ##################
    
    def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                       The action-values computed by an action-value network.              
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """

    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = action_values / tau
    # Compute the maximum preference across the actions
    max_preference = np.max(preferences, axis=1)

    
    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting 
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))
    

    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = np.exp(preferences-reshaped_max_preference)
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)

    
    
    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting 
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    
 
    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    
    
    # squeeze() removes any singleton dimensions. It is used here because this function is used in the 
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in 
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    return action_probs



def get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau):
    """
    Args:
        states (Numpy array): The batch of states with the shape (batch_size, state_dim).
        next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
        actions (Numpy array): The batch of actions with the shape (batch_size,).
        rewards (Numpy array): The batch of rewards with the shape (batch_size,).
        discount (float): The discount factor.
        terminals (Numpy array): The batch of terminals with the shape (batch_size,).
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                        and particularly, the action-values at the next-states.
    Returns:
        The TD errors (Numpy array) for actions taken, of shape (batch_size,)
    """
    
    # Note: Here network is the latest state of the network that is getting replay updates. In other words, 
    # the network represents Q_{t+1}^{i} whereas current_q represents Q_t, the fixed network used for computing the 
    # targets, and particularly, the action-values at the next-states.
    
    # Compute action values at next states using current_q network
    # Note that q_next_mat is a 2D array of shape (batch_size, num_actions)

    q_next_mat = current_q.get_action_values(next_states)

    
    # Compute policy at next state by passing the action-values in q_next_mat to softmax()
    # Note that probs_mat is a 2D array of shape (batch_size, num_actions)

    probs_mat = softmax(q_next_mat, tau=tau)
    
    # Compute the estimate of the next state value, v_next_vec.
    # Hint: sum the action-values for the next_states weighted by the policy, probs_mat. Then, multiply by
    # (1 - terminals) to make sure v_next_vec is zero for terminal next states.
    # Note that v_next_vec is a 1D array of shape (batch_size,)

    v_next_vec = np.sum(probs_mat*q_next_mat, axis=1)*(1-terminals)
    
    # Compute Expected Sarsa target
    # Note that target_vec is a 1D array of shape (batch_size,)

    target_vec = rewards + (discount*v_next_vec)

    # Compute action values at the current states for all actions using network
    # Note that q_mat is a 2D array of shape (batch_size, num_actions)

    q_mat = network.get_action_values(states)

    # Batch Indices is an array from 0 to the batch size - 1. 
    batch_indices = np.arange(q_mat.shape[0])

    # Compute q_vec by selecting q(s, a) from q_mat for taken actions
    # Use batch_indices as the index for the first dimension of q_mat
    # Note that q_vec is a 1D array of shape (batch_size)

    q_vec = q_mat[batch_indices, actions]

    # Compute TD errors for actions taken
    # Note that delta_vec is a 1D array of shape (batch_size)

    delta_vec = target_vec - q_vec

    return delta_vec

#############      testing softmax and td error                   ##################

data = np.load("asserts/get_td_error_1.npz", allow_pickle=True)

states = data["states"]
next_states = data["next_states"]
actions = data["actions"]
rewards = data["rewards"]
discount = data["discount"]
terminals = data["terminals"]
tau = 0.001

network_config = {"state_dim": 8,
                  "num_hidden_units": 512,
                  "num_actions": 4
                  }

network = ActionValueNetwork(network_config)
network.set_weights(data["network_weights"])

current_q = ActionValueNetwork(network_config)
current_q.set_weights(data["current_q_weights"])

delta_vec = get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)
answer_delta_vec = data["delta_vec"]

## Test Code for get_td_error() ##

# NOTE: The test below is limited in scope. Additional tests are used in the autograder, so it is recommended 
# to test your implementations more carefully for correctness.

data = np.load("asserts/get_td_error_1.npz", allow_pickle=True)

states = data["states"]
next_states = data["next_states"]
actions = data["actions"]
rewards = data["rewards"]
discount = data["discount"]
terminals = data["terminals"]
tau = 0.001

network_config = {"state_dim": 8,
                  "num_hidden_units": 512,
                  "num_actions": 4
                  }

network = ActionValueNetwork(network_config)
network.set_weights(data["network_weights"])

current_q = ActionValueNetwork(network_config)
current_q.set_weights(data["current_q_weights"])

delta_vec = get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)
answer_delta_vec = data["delta_vec"]

## Test Code for get_td_error() ##

# NOTE: The test below is limited in scope. Additional tests are used in the autograder, so it is recommended 
# to test your implementations more carefully for correctness.

data = np.load("asserts/get_td_error_1.npz", allow_pickle=True)

states = data["states"]
next_states = data["next_states"]
actions = data["actions"]
rewards = data["rewards"]
discount = data["discount"]
terminals = data["terminals"]
tau = 0.001

network_config = {"state_dim": 8,
                  "num_hidden_units": 512,
                  "num_actions": 4
                  }

network = ActionValueNetwork(network_config)
network.set_weights(data["network_weights"])

current_q = ActionValueNetwork(network_config)
current_q.set_weights(data["current_q_weights"])

delta_vec = get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)
answer_delta_vec = data["delta_vec"]

#############     end of testing softmax and td error                   ##################

#############     agent class def                ##################

### Work Required: Yes. Fill in code in agent_step and agent_end (~7 Lines).
class Agent():
    def __init__(self):
        self.name = "expected_sarsa_agent"
        
    # Work Required: No.
    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer, 
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'], 
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config['network_config'])
        self.optimizer = Adam(self.network.layer_sizes, agent_config["optimizer_config"])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']
        
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0
        self.episode_steps = 0

    # Work Required: No.
    def policy(self, state):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action. 
        """
        action_values = self.network.get_action_values(state)
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        return action

    # Work Required: No.
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    # Work Required: Yes. Fill in the action selection, replay-buffer update, 
    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        
        self.sum_rewards += reward
        self.episode_steps += 1

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = np.array([state])

        # Select action
        ### START CODE HERE (~1 Line)
        action = self.policy(state)
        ### END CODE HERE
        
        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments

        ### START CODE HERE (~1 Line)
        self.replay_buffer.append(self.last_state,self.last_action, reward, 0, state)
        ### END CODE HERE
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                
                # Call optimize_network to update the weights of the network (~1 Line)
                ### START CODE HERE
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)
                ### END CODE HERE
                
        # Update the last state and last action.
        ### START CODE HERE (~2 Lines)
        self.last_state = state
        self.last_action = action
        ### END CODE HERE
        
        return action

    # Work Required: Yes. Fill in the replay-buffer update and
    # update of the weights using optimize_network (~2 lines).
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        
        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments
        
        ### START CODE HERE (~1 Line)
        self.replay_buffer.append(self.last_state,self.last_action, reward, 1, state)
        ### END CODE HERE
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                
                # Call optimize_network to update the weights of the network
                ### START CODE HERE (~1 Line)
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)
                ### END CODE HERE
                
        
    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

#############     end of class for def of agent                ##################

#############     test for agent                   ##################

## Test Code for agent_step() ## 

# NOTE: The test below is limited in scope. Additional tests are used in the autograder, so it is recommended 
# to test your implementations more carefully for correctness.

agent_info = {
             'network_config': {
                 'state_dim': 8,
                 'num_hidden_units': 256,
                 'num_hidden_layers': 1,
                 'num_actions': 4
             },
             'optimizer_config': {
                 'step_size': 3e-5, 
                 'beta_m': 0.9, 
                 'beta_v': 0.999,
                 'epsilon': 1e-8
             },
             'replay_buffer_size': 32,
             'minibatch_sz': 32,
             'num_replay_updates_per_step': 4,
             'gamma': 0.99,
             'tau': 1000.0,
             'seed': 0}

# Initialize agent
agent = Agent()
agent.agent_init(agent_info)

# load agent network, optimizer, replay_buffer from the agent_input_1.npz file
input_data = np.load("asserts/agent_input_1.npz", allow_pickle=True)
agent.network.set_weights(input_data["network_weights"])
agent.optimizer.m = input_data["optimizer_m"]
agent.optimizer.v = input_data["optimizer_v"]
agent.optimizer.beta_m_product = input_data["optimizer_beta_m_product"]
agent.optimizer.beta_v_product = input_data["optimizer_beta_v_product"]
agent.replay_buffer.rand_generator.seed(int(input_data["replay_buffer_seed"]))
for experience in input_data["replay_buffer"]:
    agent.replay_buffer.buffer.append(experience)

# Perform agent_step multiple times
last_state_array = input_data["last_state_array"]
last_action_array = input_data["last_action_array"]
state_array = input_data["state_array"]
reward_array = input_data["reward_array"]

for i in range(5):
    agent.last_state = last_state_array[i]
    agent.last_action = last_action_array[i]
    state = state_array[i]
    reward = reward_array[i]
    
    agent.agent_step(reward, state)
    
    # Load expected values for last_state, last_action, weights, and replay_buffer 
    output_data = np.load("asserts/agent_step_output_{}.npz".format(i), allow_pickle=True)
    answer_last_state = output_data["last_state"]
    answer_last_action = output_data["last_action"]
    answer_updated_weights = output_data["updated_weights"]
    answer_replay_buffer = output_data["replay_buffer"]

## Test Code for agent_end() ## 

# NOTE: The test below is limited in scope. Additional tests are used in the autograder, so it is recommended 
# to test your implementations more carefully for correctness.

agent_info = {
             'network_config': {
                 'state_dim': 8,
                 'num_hidden_units': 256,
                 'num_hidden_layers': 1,
                 'num_actions': 4
             },
             'optimizer_config': {
                 'step_size': 3e-5, 
                 'beta_m': 0.9, 
                 'beta_v': 0.999,
                 'epsilon': 1e-8
             },
             'replay_buffer_size': 32,
             'minibatch_sz': 32,
             'num_replay_updates_per_step': 4,
             'gamma': 0.99,
             'tau': 1000,
             'seed': 0
             }

# Initialize agent
agent = Agent()
agent.agent_init(agent_info)

# load agent network, optimizer, replay_buffer from the agent_input_1.npz file
input_data = np.load("asserts/agent_input_1.npz", allow_pickle=True)
agent.network.set_weights(input_data["network_weights"])
agent.optimizer.m = input_data["optimizer_m"]
agent.optimizer.v = input_data["optimizer_v"]
agent.optimizer.beta_m_product = input_data["optimizer_beta_m_product"]
agent.optimizer.beta_v_product = input_data["optimizer_beta_v_product"]
agent.replay_buffer.rand_generator.seed(int(input_data["replay_buffer_seed"]))

for experience in input_data["replay_buffer"]:
    agent.replay_buffer.buffer.append(experience)

# Perform agent_step multiple times
last_state_array = input_data["last_state_array"]
last_action_array = input_data["last_action_array"]
state_array = input_data["state_array"]
reward_array = input_data["reward_array"]

for i in range(5):
    agent.last_state = last_state_array[i]
    agent.last_action = last_action_array[i]
    reward = reward_array[i]
    
    agent.agent_end(reward)

    # Load expected values for last_state, last_action, weights, and replay_buffer 
    output_data = np.load("asserts/agent_end_output_{}.npz".format(i), allow_pickle=True)
    answer_updated_weights = output_data["updated_weights"]
    answer_replay_buffer = output_data["replay_buffer"]

#############     end of tests for agent                  ##################

def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    
    rl_glue = RLGlue(environment, agent)
        
    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"], 
                                 experiment_parameters["num_episodes"]))

    env_info = {}

    agent_info = agent_parameters

    # one agent setting
    for run in range(1, experiment_parameters["num_runs"]+1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)
        
        for episode in tqdm(range(1, experiment_parameters["num_episodes"]+1)):
            # run episode
            rl_glue.rl_episode(experiment_parameters["timeout"])
            
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward
    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save("results/sum_reward_{}".format(save_name), agent_sum_reward)
    shutil.make_archive('results', 'zip', 'results')

# Run Experiment

# Experiment parameters
experiment_parameters = {
    "num_runs" : 1,
    "num_episodes" : 300,
    # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
    # some number of timesteps. Here we use the default of 1000.
    "timeout" : 1000
}

# Environment parameters
environment_parameters = {}

current_env = LunarLanderEnvironment

# Agent parameters
agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'num_hidden_units': 256,
        'num_actions': 4
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9, 
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 50000,
    'minibatch_sz': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}
current_agent = Agent

# run experiment
run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)