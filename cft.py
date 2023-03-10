import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit, vmap
from functools import partial
from flax import struct


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
    
    

