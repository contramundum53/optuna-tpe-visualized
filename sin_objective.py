import numpy as np

def objective_func(x):
    # return 0.03 * x ** 2 + 10 * np.sin(1 * x) + 10
    return np.abs(x) - 10 * np.cos(1 * x) + 10
x_range = (-10, 50)
def objective(trial):
    x = trial.suggest_uniform('x', x_range[0], x_range[1])
    # x = trial.suggest_categorical('x', [-5, 0, 5, 10, 15, 20])
    y = trial.suggest_categorical('y', ["a", "b", "c"])
    return objective_func(x)