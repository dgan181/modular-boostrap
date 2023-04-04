from cft import *
from newton import *

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, grad, jacfwd
import os
PATH = os.getcwd()

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import optax
from tqdm import tqdm

from scipy import optimize



jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=20)

#random number generator seed = 1
rng = random.PRNGKey(1)

# Define a model
c = 4 # <- change this to generate values for a different 


list_losses = []
list_params = []
list_guesses = []

# Initialisations 1
deltas = jnp.array([2.13, 3.4, 5.1]) # starting guess for deltas
a_mus = jnp.ones(deltas.shape, dtype = float) # starting guess for a_mus is an array of ones
params = jnp.array([deltas,a_mus])
params = params.flatten()

list_guesses.append(params)

keys = random.split(rng, 2)
nr_primal = partial(newton_rap,primal) # select which newton raphson to use: primal in this case

print("Subroutine 1")
# sub-routine 1 to get the first few parameters
for run in range(10):
    print(f" run {run} with number of deltas = {run + 3}")
    # check whether params are optimal by using newton raphson
    params, *others = nr_primal(x0 = params, c=c)
    if others[-1]: # abort condition
        print("Encountered NaN: aborting")
        break
    list_params.append(params)
    # generate guesses for m+1 deltas using the previous m deltas
    new_deltas, keys = initial_guess_generation(params, keys)
    a_mus = jnp.ones(new_deltas.shape, dtype = float)
    params = jnp.array([new_deltas,a_mus]).flatten()
    list_guesses.append(params)


print("Subroutine 2")
# sub-routine 2 to get the higher order parameters with a more sophisticated guess generation scheme

#Initialisations 2
deltas_nm1 = list_params[-2][:len(list_params[-2])//2] # delta n - 1
deltas_n = deltas_nm1 # delta n
deltas_g = list_params[-1][:len(list_params[-1])//2] # delta guess
adam = optax.adam(learning_rate=0.2)

a_mus_g = list_params[-1][len(list_params[-2])//2+1:]

h = len(deltas_g)
nr_primal_normal = partial(newton_rap,primal_normal)
for p in range(20):
    print(f"run {p} with number of deltas = {h}")
    
    # guess generation for a_mus -> Optimise over a_mus with fixed deltas 
    a_mus_g, losses, _ = rhos_optim(a_mus_g, deltas_g, c, opt_fn=adam.update, opt_state=adam.init(a_mus_g), steps=2000)

    params = jnp.array([deltas_g,a_mus_g])
    params = params.flatten()

    # newton method
    params, *others = nr_primal_normal(params,c=c)

    if others[-1]: # abort condition
        print("Encountered NaN: aborting")
        break 

    list_params.append(params)
    list_losses.append(primal_normal(params,c=c))

    deltas_nm1 = deltas_n
    deltas_n = lax.dynamic_slice(params, (0,),(h,))
    a_mus_n = lax.dynamic_slice(params, (h,),(h,))
    
    # Generate guess for delta_{n + 1} using delta_{n} and deltas_{n - 1}
    deltas_g = guess_generator(deltas_nm1, deltas_n, fixed_point=1) 
    a_mus_g = jnp.ones(deltas_g.shape, dtype = float) 
    a_mus_g = a_mus_g.at[0:h].set(a_mus_n)

    list_guesses.append(deltas_g)
    h += 1

print("Saving parameters in directory spinless-parameters")
jnp.save(f"{PATH}/spinless-parameters/c-{c}",np.array(list_params, dtype=object))

# To load `jnp.load(/path/to/.npy)`