#%%
import optuna
from _tpe.sampler import TPESampler
from _tpe.sampler import default_gamma, default_weights
import numpy as np
import matplotlib.pyplot as plt

def objective_func(x):
    # return 0.03 * x ** 2 + 10 * np.sin(1 * x) + 10
    return np.abs(x) - 10 * np.cos(1 * x) + 10

x_range = (-10, 50)
def objective(trial):
    x = trial.suggest_uniform('x', x_range[0], x_range[1])
    return objective_func(x)

X = np.linspace(x_range[0], x_range[1], 10000)
Y = np.array([objective_func(x) for x in X])

plt.plot(X, Y)
# plt.show()
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
        # print(content)
        log_objects.append(content)    



    study = optuna.create_study(sampler=TPESampler(
                                        consider_prior = True,
                                        prior_weight = 1.0,
                                        consider_magic_clip = False,#True,
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
