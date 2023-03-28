import jax
from cft import *
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import optax
from jax import jit, vmap, lax, grad, jacfwd

from functools import partial

from scipy import optimize

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=20)

from matplotlib.pyplot import cm

from jax.config import config
config.update("jax_debug_nans", True)

from jaxopt import ScipyRootFinding
from scipy import optimize

c = 12
x = -(c-1)/12

def primal(params: jnp.array) -> jnp.array:
    """
    Crossing equation with unnormalised rhos    
    
    Args:
        params (jnp.array) [deltas, rhos]
    Returns:
        crossing_equation -> float, crossing equation loss
    """
    h = len(params)//2
    deltas = lax.dynamic_slice(params, (0,),(h,)) # split deltas and rhos
    rhos = lax.dynamic_slice(params, (h,),(h,))
    deltas = deltas + x
    lps = laguerre_deltas(2*h -1,deltas) # calculate laguerre at different deltas
    lp0 = laguerre_at_0(2*h-1, x)# calculate laguerre at zero
    rho_lps = jnp.sum(rhos * lps, axis = 1) # rho * lps
    crossing_equation = (1 + rho_lps/lp0) # lp0 + \Sigma rho* lps = 0
    return  crossing_equation


def newton_rap(primal, x0: jnp.array, tol=1e-10, max_iter=2000) -> (jnp.array,jnp.array,float,bool):
    """
    Finds a solution to `primal` using the Newton-Raphson method. 
    Written with jax while loop.

    Args:
        primal (function): function to approximate a solution of
        x0 (jnp.array): Initial guess for the solution vector.
        tol (float): Tolerance for the norm of the residual vector. Default is 1e-6.
        max_iter (int): Maximum number of iterations. Default is 100.

    Returns:
        x (jnp.array,jnp.array,float,bool): Converging parameters (deltas and rhos), number of steps, converges?
    """

    f = primal

    Jf = jit(jacfwd(f))

    @jit
    def newton_step(val):
        x, r, n, _ = val # x, f(x), step_number, abort_state
        n = n + 1 
        J = Jf(x)
        r = f(x)
        dx = jnp.linalg.solve(J, -r)
        x1 = x + dx
        abort = jnp.any(jnp.isnan(x1)) # abort if nan is encountered
        return (jnp.where(abort, x, x1), r , n, abort)
    
    def cond(val):
        x, r, n, abort = val
        return (~abort) & (jnp.max(jnp.abs(r)) > tol) & ( n< max_iter) # abort if nan is encountered

    
    return  lax.while_loop(cond, newton_step, (x0, jnp.ones(x0.shape), 0.0, False))  

@jit
def primal_normal(params):
    """
    Crossing equation with normalised rhos 
    
    params -> [deltas, rhos]
    lps -> laguerre polynomials at deltas
    lp0 -> laguerre at 0
    Crossing equation with normalised rhos -> rhos * \sqrt(max(lps))
    """
    h = len(params)//2
    deltas = lax.dynamic_slice(params, (0,),(h,))
    rhos = lax.dynamic_slice(params, (h,),(h,))
    deltas = deltas + x
    lps = laguerre_deltas(2*h -1,deltas)
    maxl = jnp.max(jnp.abs(lps), axis=0)
    lps = lps/jnp.sqrt(maxl)
    lp0 = laguerre_at_0(2*h-1, x)
    rho_lps = jnp.sum(rhos * lps, axis = 1)
    crossing_equation = (1 + rho_lps/lp0)
    return  crossing_equation

@jit
def primal_normal_optim(rhos,deltas):
    """
    Mean squared loss of the crossing equation
    Args:
        rhos (jpn.array(float)) rhos
        deltas (jpn.array(float)) deltas
    Returns:
        Mean square loss (float)
    """
    
    h = len(deltas)
#     deltas = lax.dynamic_slice(params, (0,),(h,))
#     rhos = lax.dynamic_slice(params, (h,),(h,))
    deltas = deltas + x
    lps = laguerre_deltas(2*h -1,deltas)
    maxl = jnp.max(jnp.abs(lps), axis=0)
    lps = lps/jnp.sqrt(maxl)
    lp0 = laguerre_at_0(2*h-1, x)
    rho_lps = jnp.sum(rhos * lps, axis = 1)
    crossing_equation = (1 + rho_lps/lp0)
    return  jnp.linalg.norm(crossing_equation)/h