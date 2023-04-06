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

import hydra
import optax
from tqdm import tqdm
from omegaconf import DictConfig

@hydra.main(version_base='1.2', config_path="conf", config_name="cft")
def main(cfg: DictConfig):
    jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(precision=cfg.jax_settings.precision)
    
    #random number generator seed = 1
    rng = random.PRNGKey(cfg.random_seed)
    
    # Define a model
    c = cfg.model.central_charge # <- change this to generate values for a different 
    
    print(f"central charge: c = {c}")
    
    list_losses = []
    list_params = []
    
    # Initialisations at step 1
    deltas = hydra.utils.instantiate(cfg.model.initializations.deltas) # starting guess for deltas
    a_mus = jnp.ones(deltas.shape, dtype = float) # starting guess for a_mus is an array of ones
    params = jnp.array([deltas,a_mus])
    params = params.flatten()
    
    keys = random.split(rng, 2)
    nr_primal = partial(newton_rap,primal) # select which newton raphson to use: primal in this case
    
    print("Subroutine 1")
    # sub-routine 1 to get the first few parameters
    run = 0
    while True:
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
        run += 1
    
    assert len(list_params)>1, "Subroutine 1 failed to generate any parameters. Subroutine 2 cannot proceed."
    
    print("Subroutine 2")
    # sub-routine 2 to get the higher order parameters with a more sophisticated guess generation scheme, but with at least two known parameters.
    
    #Initialisations 2
    deltas_nm1 = list_params[-2][:len(list_params[-2])//2] # delta n - 1
    deltas_n = deltas_nm1 # delta n
    deltas_g = list_params[-1][:len(list_params[-1])//2] # delta guess
    opt = optax.adam(cfg.training.learning_rate)
    
    a_mus_g = list_params[-1][len(list_params[-2])//2+1:]
    
    h = len(deltas_g)
    nr_primal_normal = partial(newton_rap,primal_normal)
    
    h_offset = jnp.copy(h)

    print(opt, type(opt))
    while True:
        print(f"run {h-h_offset} with number of deltas = {h}")
        
        # guess generation for a_mus -> Optimise over a_mus with fixed deltas 
        # mean square loss for the primal function over a_mus
        opt_state =  opt.init(a_mus_g)
        a_mus_g, losses, _ = rhos_optim(a_mus_g, deltas_g, c, opt_fn=opt.update, opt_state= opt_state, steps=cfg.training.optimizer_steps)
    
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
    
        h += 1
    
    if not (len(list_params) == 0):
        savedir = f"{PATH}/spinless-parameters/c-{c}"
        save = True
        if os.path.isdir(savedir):
            saved_params = jnp.load(savedir+".npy", allow_pickle=True)
            if len(saved_params) >= len(list_params):
                save = False
        if save:
            print("Saving parameters in directory spinless-parameters")
            jnp.save(f"{PATH}/spinless-parameters/c-{c}",np.array(list_params, dtype=object))
        else:
            print("Nothing new to save")

if __name__ == "__main__":
    main()
