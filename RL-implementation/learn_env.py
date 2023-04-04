from stable_baselines3 import PPO
from bootstrap_env import BootstrapENV
import os
import time
import sys

# dim = int(sys.argv[1])
# i = int(sys.argv[2])

# print(f"Learning dim:{dim}, i:{i}")
number = 1
which = 1
model_train=f"models/{number}/{which}"

logdir = f"logs/{int(time.time())}/"

env = BootstrapENV()
env.reset()

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(model_train):
    models_dir = f"models/{int(time.time())}/"
    os.makedirs(models_dir)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    iters = 0
else:
    model = PPO.load(model_train, env = env, tensorboard_log=logdir )
    models_dir = model_train
    iters = which


TIMESTEPS = 50000

while iters < 200:
    iters += 1 
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")