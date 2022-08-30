import matplotlib.pyplot as plt
import numpy as np

import pickle
import sys, os

log_file = sys.argv[1]

max_iter = None
if len(sys.argv) > 2:
    max_iter = int(sys.argv[2])

out_dir = None
if len(sys.argv) > 3:
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

log_data = pickle.load(open(log_file, "rb"))


def objective_func(x):
    # return 0.03 * x ** 2 + 10 * np.sin(1 * x) + 10
    return np.abs(x) - 10 * np.cos(1 * x) + 10

x_range = (-10, 50)

X = np.linspace(x_range[0], x_range[1], 10000)
Y = np.array([objective_func(x) for x in X])

for log_frame in log_data:
    if max_iter is not None and log_frame["trial_number"] > max_iter:
        break

    if log_frame["func"] == "sample_independent":
        print(log_frame)
        param_name = log_frame["param_name"]
        dist = log_frame["param_distribution"]
        bounds = (dist.low, dist.high)

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
        # prob_low = 1 / (1 + np.exp((np.log(len(indices_above)) + log_above_dist) - (np.log(len(indices_below)) + log_below_dist)))
        prob_low = prob_low_func(X)

        samples_below = log_frame["samples_below"][param_name]
        ret = log_frame["ret"]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_title(f"TPE (param:'{param_name}', number:{trial_number})")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Value")
        ax.set_xlim(*bounds)
        

        ax.plot(X, Y, color="gray", alpha=0.3, label="Objective")
        ax.plot(values[indices_above], scores[indices_above], label="Above", marker=".", linestyle="none", color="red", alpha=0.3)
        ax.plot(values[indices_below], scores[indices_below], label="Below", marker=".", linestyle="none", color="blue", alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.plot(X, above_dist / dist_max, color="red", linestyle=":", alpha=0.5, label="Above dist.")
        ax2.plot(X, below_dist / dist_max, color="blue", linestyle=":", alpha=0.5, label="Below dist.")
        ax2.plot(X, prob_low, color="green", label="Prob. of below")

        ax2.plot(samples_below, prob_low_func(samples_below), marker=".", linestyle="none", color="black", alpha=0.5, label="Samples", markersize=5)
        ax2.plot([ret], prob_low_func(np.array([ret])), marker=".", linestyle="none", color="red", label="ret", markersize=10)

        ax.legend()
        ax2.legend()
        if out_dir is not None:
            fig.savefig(f"{out_dir}/{param_name}_{trial_number}.png")
        else:
            plt.show()
        