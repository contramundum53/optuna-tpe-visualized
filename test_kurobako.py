#%%
import optuna
from _tpe.sampler import TPESampler
from _tpe.sampler import default_gamma, default_weights
import numpy as np
import matplotlib.pyplot as plt
import kurobako_problem


hpobench = kurobako_problem.KurobakoProblem({"hpobench": {"dataset": "/Users/yunzhuowang/optuna/fcnet_tabular_benchmarks/fcnet_naval_propulsion_data.hdf5"}}, 0)

def objective(trial):
    activation_fn_1 = trial.suggest_categorical('activation_fn_1', [0, 1])
    activation_fn_2 = trial.suggest_categorical('activation_fn_2', [0, 1])
    batch_size = trial.suggest_int('batch_size', 0, 3)
    dropout_1 = trial.suggest_int('dropout_1', 0, 2)
    dropout_2 = trial.suggest_int('dropout_2', 0, 2)
    init_lr = trial.suggest_int('init_lr', 0, 5)
    lr_schedule = trial.suggest_categorical('lr_schedule', [0, 1])
    n_units_1 = trial.suggest_int('n_units_1', 0, 5)
    n_units_2 = trial.suggest_int('n_units_2', 0, 5)
    return hpobench([activation_fn_1, activation_fn_2, batch_size, dropout_1, dropout_2, init_lr, lr_schedule, n_units_1, n_units_2])


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

    study.optimize(objective, n_trials=100)

    pickle.dump(log_objects, open(f"{log_dir}/log_{seed}.pkl", "wb"))

# %%
