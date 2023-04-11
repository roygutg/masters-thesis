import pandas as pd
import matplotlib.pyplot as plt
from utils import scatter, ROOT

plt.style.use('comdepri.mplstyle')

data = pd.read_csv(rf"{ROOT}\data\vanillaMeasures.csv")
print(f"N = {len(data)}")

y_dict = {"Gallery originality": data["Gallery Orig"], "% Gallery uniqueness": data["% Galleries Unique"],
          "Out-of-the-boxness": 1 - data["% clusters in GC"], "Exploration efficiency": data["exp optimality"],
          "Exploitation efficiency": data["scav optimality"], "Path length": data["Total # moves"]}

fig, axs = plt.subplots(2, 3, sharex=True, figsize=(17, 10))
for ax, (y_label, y) in zip(axs.flat, y_dict.items()):
    data_dict = {"Self-avoidance": data["self avoidance"], y_label: y}
    scatter(data_dict, ax)

plt.savefig(rf"{ROOT}\figs\vanilla_correlations.pdf", bbox_inches="tight")
plt.show()
