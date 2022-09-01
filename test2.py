#%%
import optuna
from _tpe.sampler import TPESampler
from _tpe.sampler import default_gamma, default_weights
import numpy as np
import matplotlib.pyplot as plt
import kurobako_problem


import numpy as np

def objective(trial):
    x = trial.suggest_categorical('x', [0, 1, 2, 3])
    return x

#%%
import pickle 
import datetime
import os
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = f"log-{timestamp}"
os.makedirs(log_dir, exist_ok=True)

for seed in range(10):
    log_objects = []

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
                                        gamma = default_gamma,
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
