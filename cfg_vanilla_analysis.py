import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import scatter, ROOT

plt.style.use('comdepri.mplstyle')

data = pd.read_csv(rf"{ROOT}\data\vanillaMeasures.csv")
print(f"N = {len(data)}")

orig_dict = {"Originality": data["Gallery Orig"], "% unique products": data["% Galleries Unique"],
             "% novel categories\ndiscovered": 1 - data["% clusters in GC"]}
efficiency_dict = {"Exploration efficiency": data["exp optimality"],
                   "Exploitation efficiency": data["scav optimality"]}

for y_dict, y_name in zip((orig_dict, efficiency_dict), ("originality", "efficiency")):
    n_subplots = len(y_dict)
    fig, axs = plt.subplots(1, n_subplots, sharex=True, figsize=(2 + n_subplots * 5, 5))
    for ax, (y_label, y) in zip(tqdm(axs.flat, desc="Vanilla vars", leave=False), y_dict.items()):
        data_dict = {"Self-avoidance": data["self avoidance"], y_label: y}
        scatter(data_dict, ax)

    plt.savefig(rf"{ROOT}\figs\{y_name}_correlations.pdf", bbox_inches="tight")
    plt.show()
