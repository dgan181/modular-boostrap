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

# from jax.config import config
# config.update("jax_debug_nans", True)

# from jaxopt import ScipyRootFinding
from scipy import optimize


def primal(params: jnp.array, c: float) -> jnp.array:
    """
    Crossing equation with unnormalised rhos    
    
    Args:
        params (jnp.array) [deltas, rhos]
    Returns:
        crossing_equation -> float, crossing equation loss
    """
    x = -(c-1)/12
    h = len(params)//2
    deltas = lax.dynamic_slice(params, (0,),(h,)) # split deltas and rhos
    rhos = lax.dynamic_slice(params, (h,),(h,))
    deltas = deltas + x
    lps = laguerre_deltas(2*h -1,deltas) # calculate laguerre at different deltas
    lp0 = laguerre_at_0(2*h-1, x)# calculate laguerre at zero
    rho_lps = jnp.sum(rhos * lps, axis = 1) # rho * lps
    crossing_equation = (1 + rho_lps/lp0) # lp0 + \Sigma rho* lps = 0
    return  crossing_equation


def newton_rap(primal, x0: jnp.array, c: float, tol=1e-10, max_iter=2000):
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

    f = partial(primal, c=c)

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
def primal_normal(params, c):
    """
    Crossing equation with normalised rhos 
    
    params -> [deltas, rhos]
    lps -> laguerre polynomials at deltas
    lp0 -> laguerre at 0
    Crossing equation with normalised rhos -> rhos * \sqrt(max(lps))
    """
    x = -(c-1)/12
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
def primal_normal_optim(rhos,deltas,c):
    """
    Mean squared loss of the crossing equation
    Args:
        rhos (jpn.array(float)) rhos
        deltas (jpn.array(float)) deltas
    Returns:
        Mean square loss (float)
    """
    x = -(c-1)/12
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

def rhos_optim(rhos, deltas, c, opt_fn, opt_state, steps=100):
    """ Guess rhos, given a good guess for deltas

        Params:
        rho - initial values for rhos (arbitrary)
        deltas - intitialised deltas (fixed)
        opt_fn - optimization function e.g. Adam
        opt_state - initial state

        Returns:
        rhos, losses,  state

    """
    x = -(c-1)/12
    h = len(deltas)
    deltas = deltas + x
    @jit
    def loss_function(rhos): # same as primal_normal
        lps = laguerre_deltas(2*h -1,deltas)
        maxl = jnp.max(jnp.abs(lps), axis=0)
        lps = lps/jnp.sqrt(maxl)
        lp0 = laguerre_at_0(2*h-1, x)
        rho_lps = jnp.sum(rhos * lps, axis = 1)
        crossing_equation = (1 + rho_lps/lp0)
        return jnp.mean((crossing_equation)**2) # mean squared loss

    losses = []
    for _ in tqdm(range(steps)): # optimiser
        loss, grads = jax.value_and_grad(loss_function)(rhos) # loss and gradients calulated 
        updates, opt_state = opt_fn(grads, opt_state) 
        rhos += updates # update rho
        losses.append(loss) 

    return  rhos, jnp.stack(losses), opt_state

def dilation(dnm1,dn, fixed_point = 1, some_point = 0):
    """Calculate scaling factor \epsilon for self-similar function: 
        f(\lambda x) = \lambda^{\epsilon} f(x)
        
        Args:
        dnm1: x
        dn: \lambda x
        fixed_point:
        some_point:
        
        Returns:
        epsilons: \epsilon
    """
    h =  len(dnm1) - some_point if len(dnm1) - some_point> 0 else len(dnm1)
    lambda_= (h+1 - fixed_point)/(h - fixed_point)
    epsilon = jnp.log((dn[h+1]-dn[fixed_point-1])/(dnm1[h] - dnm1[fixed_point-1]))/jnp.log(lambda_) 
    return epsilon

@jit
def initial_guess_generation(params, keys = random.split(random.PRNGKey(0))):
    #check whethere params converges
    h = len(params)//2
    r = h-1
    deltas = lax.dynamic_slice(params, (0,),(h,))
    scale = (deltas[r] - deltas[0])/(r)
    keys = random.split(keys[1],2)
    new_deltas = (scale * (jnp.arange(h+1)) + deltas[0] ) + random.uniform(keys[0], jnp.arange(h+1).shape) * 0.1/(h+1)    
    return new_deltas, keys

def guess_generator(dnm1, dn, fixed_point=1):
    """generate guess dnp1 from dnm1 and dn

        
        Args:
        dnm1: x
        dn: \lambda x
        fixed_point:
        
        Returns:
        guess: dnp1
    """
    
    epsilon = dilation(dnm1, dn, fixed_point,0)
    no_points = 1000
    x_range = jnp.linspace(1.0,int(len(dn)),(int(len(dn))- 1)*no_points)
    
    x_discrete = jnp.arange(1, int(len(dn)) + 1, dtype = float)
    
    yn = jnp.interp(x_range, x_discrete, dn)
    
    lambda_ = (len(dn))/(len(dn) - 1)
    
    scale = lambda_ ** epsilon
    
    ynp1 = (yn - yn[0]) * scale + yn[0]
    sample_rate = int((no_points-1)/ lambda_)
    # Correction added to guess
    # cors = jnp.array([0.007866155585121012, 0.00954485954845927 , 0.014117004802336118,
    #    0.023951187818932042, 0.046797643955721294, 0.10137571941656    ])
    guess = ynp1[::sample_rate]
    # guess= guess.at[-6:].add(cors)
    return guess