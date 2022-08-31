import matplotlib.pyplot as plt
import numpy as np

import pickle
import sys, os
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

log_file = sys.argv[1]

max_iter = None
if len(sys.argv) > 2:
    max_iter = int(sys.argv[2])

out_dir = None
if len(sys.argv) > 3:
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

log_data = pickle.load(open(log_file, "rb"))

objective_func_data = None

for log_frame in log_data:
    if max_iter is not None and "trial_number" in log_frame and log_frame["trial_number"] > max_iter:
        break
    
    print(log_frame)
    if log_frame["log_type"] == "objective_data":
        objective_func_data = log_frame["data"]
    elif log_frame["log_type"] == "sample_independent":
        param_name = log_frame["param_name"]
        dist = log_frame["param_distribution"]
        if isinstance(dist, FloatDistribution):
            X = np.array([dist.to_internal_repr(x) for x in np.linspace(dist.low, dist.high, 1000)])
            is_log = dist.log
        elif isinstance(dist, IntDistribution):
            X = np.array([dist.to_internal_repr(x) for x in range(dist.low, dist.high+1)])
            is_log = dist.log
        elif isinstance(dist, CategoricalDistribution):
            X = np.arange(len(dist.choices))
            is_log = False
        

        values = np.array(log_frame["values"][param_name])
        scores = np.array([score[0] for step, score in log_frame["scores"]])

        trial_number = log_frame['trial_number']
        indices_below = log_frame["indices_below"]
        indices_above = log_frame["indices_above"]

        mpe_below = log_frame["mpe_below"]
        mpe_above = log_frame["mpe_above"]
        log_below_dist = mpe_below.log_pdf({param_name: X})
        log_above_dist = mpe_above.log_pdf({param_name: X})
        below_dist = np.exp(log_below_dist)
        above_dist = np.exp(log_above_dist)
        dist_max = np.max(np.concatenate([below_dist, above_dist]))

        prob_low_func = lambda x: 1 / (1 + np.exp(
            (np.log(len(indices_above)) + mpe_above.log_pdf({param_name: x})) 
                - (np.log(len(indices_below)) + mpe_below.log_pdf({param_name: x}))))
        prob_low = prob_low_func(X)

        samples_below = log_frame["samples_below"][param_name]
        ret_internal = log_frame["ret_internal"]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_title(f"TPE (param:'{param_name}', number:{trial_number})")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Value")
        if is_log:
            ax.set_xscale("log")
        

        transform = lambda X: [dist.to_external_repr(x) for x in X]



        if objective_func_data is not None and param_name in objective_func_data:
            objective_X = objective_func_data[param_name][0]
            objective_Y = objective_func_data[param_name][1]
            ax.plot(objective_X, objective_Y, color="gray", alpha=0.3, label="Objective")
        ax.plot(transform(values[indices_above]), scores[indices_above], label="Above", marker=".", linestyle="none", color="red", alpha=0.3)
        ax.plot(transform(values[indices_below]), scores[indices_below], label="Below", marker=".", linestyle="none", color="blue", alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1.1)
        ax2.plot(transform(X), above_dist / dist_max, color="red", linestyle=":", alpha=0.5, label="Above dist.")
        ax2.plot(transform(X), below_dist / dist_max, color="blue", linestyle=":", alpha=0.5, label="Below dist.")
        ax2.plot(transform(X), prob_low, color="green", label="Prob. of below")

        ax2.plot(transform(samples_below), prob_low_func(samples_below), marker="*", linestyle="none", color="black", alpha=0.5, label="Samples", markersize=5)
        ax2.plot(transform([ret_internal]), prob_low_func(np.array([ret_internal])), marker="*", linestyle="none", color="red", label="ret", markersize=10)

        ax.legend()
        ax2.legend()
        if out_dir is not None:
            fig.savefig(f"{out_dir}/{param_name}_{trial_number}.png")
        else:
            plt.show()
        