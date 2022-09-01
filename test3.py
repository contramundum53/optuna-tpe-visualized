#%%
import optuna
from _tpe.sampler import TPESampler
from _tpe.sampler import default_gamma, default_weights
import numpy as np
import matplotlib.pyplot as plt
import kurobako_problem


import numpy as np

N = 10

k = 1.0

def objective(trial):
    x = trial.suggest_categorical('x', list(range(N)))
    return float(np.random.normal(x, k * x))
    

#%%
import pickle 
import datetime
import os
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = f"log-{timestamp}"
os.makedirs(log_dir, exist_ok=True)

for seed in range(2):
    log_objects = []
    log_objects.append({
        "log_type": "objective_data",
        "data": {
            "x": (np.arange(N), np.arange(N))
        }
    })

    def logger_callback(content):
        global log_objects
        log_objects.append(content)    


    study = optuna.create_study(sampler=TPESampler(
                                        consider_prior = True,
                                        prior_weight = 1.0,
                                        consider_magic_clip = True,
                                        consider_endpoints = False,
                                        n_startup_trials = 10,
                                        n_ei_candidates = 24,
                                        gamma = lambda x: int(np.ceil(x * 0.1)),
                                        weights = default_weights,
                                        seed = seed,
                                        multivariate = False,
                                        group = False,
                                        constant_liar = False,
                                        algorithm_logger_callback = logger_callback,
                                    ),direction='minimize')

    study.optimize(objective, n_trials=1000)

    pickle.dump(log_objects, open(f"{log_dir}/log_{seed}.pkl", "wb"))

# %%
