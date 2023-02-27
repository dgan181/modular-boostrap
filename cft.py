import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit, vmap, lax
from functools import partial
from flax import struct

import numpy as np
from numpy.polynomial import Laguerre


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
def laguerre(Ls,n, x):
    """Laguerre odd recusision 
        Params:
        Ls: L_{n - 1} , L_{n + 1}
        n: n
        x: x

        Returns:
        (L_{n + 1} , L_{n + 2}) , L_{n + 2}
    """
    Lnm1, Ln = Ls
    Lnp1 = ((2 * n + 1 - x) * Ln - n * Lnm1) / (n + 1)
    Lnp2 = ((2 * (n + 1) + 1 - x) * Lnp1 - (n + 1) * Ln) / ((n + 1) + 1)
    return  (Lnp1,Lnp2), Lnp2

@jit
def laguerre_with_derivatives(Ls,n, x):
    
    Lnm1, Ln = Ls
    Lnp1 = ((2 * n + 1 - x) * Ln - n * Lnm1) / (n + 1)
    Lnp2 = ((2 * (n + 1) + 1 - x) * Lnp1 - (n + 1) * Ln) / ((n + 1) + 1)
    derLnp2 =  ((n + 1) + 1) * (Lnp2 - Lnp1) / (x)
    return  (Lnp1,Lnp2), jnp.array([Lnp2, derLnp2])

@jit
def laguerre_derivates(Ls,n, x):
    Lnm1, Ln = Ls
    Lnp1 = ((2 * n + 1 - x) * Ln - n * Lnm1) / (n + 1)
    Lnp2 = ((2 * (n + 1) + 1 - x) * Lnp1 - (n + 1) * Ln) / ((n + 1) + 1)
    derLnp2 =  ((n + 1) + 1) * (Lnp2 - Lnp1) / (x)
    return  (Lnp1,Lnp2), derLnp2   

@partial(jit, static_argnames=['order'])
def laguerre_at_x(order, x):
    """Vector with odd Laguerre polynomials
    """
    x = 4 * jnp.pi * (x)
    laguerre_x = partial(laguerre, x = x)
    ns = jnp.arange(-1, order*2, step = 2, dtype=float)
    ns = ns.at[0].add(1e-8)
    _ , lps = lax.scan(laguerre_x, init=(1e-8, 0.0), xs=ns)
    return lps

@partial(jit, static_argnames=['order'])
def laguerre_at_0(order, x):
    """Vector with odd Laguerre polynomials at delta = 0
    """
    return laguerre_at_x(order, x) - 2 * jnp.exp(-2 * jnp.pi) * laguerre_at_x(order, x + 1) + jnp.exp(-4 * jnp.pi) * laguerre_at_x(order, x + 2) 