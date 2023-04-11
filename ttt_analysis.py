import pandas as pd
import matplotlib.pyplot as plt
from utils import scatter, ROOT

plt.style.use('comdepri.mplstyle')

data = pd.read_csv(f"{ROOT}/data/ttt_data.csv")
print(f"N = {len(data)}")
ttt_vars = {"Focused search": data["shutter_zero_percent"], "Paths entropy": data["first_moves_entropy"],
            "Forcing log-likelihood": data["blocking"], "Work": data["work"]}

fig, axs = plt.subplots(2, 2, sharex=True)
for ax, (ttt_label, y) in zip(axs.flat, ttt_vars.items()):
    data_dict = {"Self-avoidance": data["self avoidance"], ttt_label: y}
    scatter(data_dict, ax)

plt.savefig(fr"{ROOT}\figs\ttt_correlations.pdf", bbox_inches="tight")
plt.show()
