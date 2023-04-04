import gym
from gym import spaces
import numpy as np
import random
import time
import copy

import jax
from jax import random, jit, vmap, lax
import jax.numpy as jnp

@jit
def q(beta):
    """ calculates nome in terms of beta such that beta*j = tau """
    return jnp.exp(-1 * 2 * np.pi * beta)

@jit
def reduced_chi_0(beta, c):
    """ calculates reduced (without the factor of eta) character of vacuum """
    return (beta**0.5) * (q(beta)**( -(c-1) / 12)) * ((1-q(beta))**2)

@jit
def reduced_chi_delta(delta, beta, c):
    """ calculates reduced (without the factor of eta) character of primary with scaling dimension delta and no spin"""
    return  (beta**0.5) * q(beta)**(delta - (c - 1) / 12) 

@jit
def reduced_partition_function_spinless(params, beta, c):
    """ calculates reduced partition function as a function of beta, c, and an array of deltas. 
        Length of deltas corresponds to trunctation length.
    """
    characters = vmap(reduced_chi_delta, in_axes=(0,None,None), out_axes=0) # defines a vectorisable map for characters
    return jnp.sum(characters(params[0], beta, c) * params[1]) + reduced_chi_0(beta,c) 

@jit
def loss_function(params, beta, c):
    identy = vmap(reduced_partition_function_spinless, in_axes=(None,0,None), out_axes=0)(params, beta, c)
    transformed = vmap(reduced_partition_function_spinless, in_axes=(None,0, None), out_axes=0)(params, 1/beta, c)

    dif = (identy-transformed)/(jnp.max(identy)-jnp.max(transformed))
    return jnp.mean((dif)**2) 

class BootstrapENV(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BootstrapENV, self).__init__()
        # 3. add 1
        # So we need a MultiDiscrete action space of self.ind_entries sets with three actions each
        self.truncation = 6
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(int(self.truncation*2),)) 
        # Example for using image as input:
        # Observations will be the independent entries of the difference of transformed and nontransformed matrices
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.truncation*2,))
        self.count = 0
        self.tol = 1e-4
        self.batch_size = 15
        self.c = 4
        self.rng = random.PRNGKey(10)
        self.keys = random.split(self.rng, 43)
        self.beta = (random.uniform(self.keys[0], (self.batch_size,)) + 0.6 )
        self.observation = random.uniform(self.keys[1], (self.truncation*2,))*(self.truncation**2)

    def step(self, action):
        self.count += 1
        self.done = False
        info = {}
        new_params = self.observation + jnp.array(action)
        self.observation = new_params
        # print(new_params)
        new_params = new_params.reshape((2, self.truncation))
        new_params = jnp.where(new_params< 0 , 0, new_params)
        loss = loss_function(new_params, self.beta, self.c)
        
        # print(loss)
        # minimum_delta = jnp.min(new_params[0])
        # negativity_punishement = jnp.sum(jnp.abs(new_params - jnp.abs(new_params)))

        loss = np.nan_to_num(loss)
        if loss > 1e5:
            loss = 1e5
        self.reward = - loss # - negativity_punishement
        if self.count % 100 ==0:
            print(f"loss = {new_params}")

        if self.count % 1000 ==0:
            print(f"loss = {loss}, reward = {self.reward}")

        if loss < self.tol:
            print(self.observation)
            self.keys = random.split(self.keys[0], 2)
            self.observation = random.uniform(self.keys[1], (self.truncation*2,))*(self.truncation*2)
            
        if self.count > 10000:
            self.done = True
            
        return self.observation, self.reward, self.done, info

    def reset(self):
        self.count = 0
        self.keys = random.split(self.keys[1], 2)
        self.observation = random.uniform(self.keys[1], (self.truncation*2,))*(self.truncation*2)
        return self.observation  # reward, done, info can't be included

    def render(self):
        pass