#%%
import json
import subprocess

class KurobakoProblem:
    def __init__(self, problem, seed = None):
        cmd = ["kurobako", "batch-evaluate", "--problem", json.dumps(problem)] + (["--seed", f"{seed}"] if seed is not None else [])
        self._subprocess = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, 
                         bufsize=0)

    def __call__(self, x, step=None):

        data = {"params": x}
        if step is not None:
            data["step"] = step
        self._subprocess.stdin.write((json.dumps(data) + "\n").encode("utf-8"))
        self._subprocess.stdin.flush()
        return json.loads(self._subprocess.stdout.readline())["values"]

    def __del__(self):
        self._subprocess.stdin.close()

# #%%

# p = KurobakoProblem({"hpobench": {"dataset": "/Users/yunzhuowang/optuna/fcnet_tabular_benchmarks/fcnet_naval_propulsion_data.hdf5"}}, 0)

# # %%
# p([1,1,1,1,1,1,1,1,1], step=50)

# # %%
# del p
# # %%
