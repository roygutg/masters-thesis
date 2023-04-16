import pandas as pd
import matplotlib.pyplot as plt
from utils import scatter, ROOT

plt.style.use('comdepri.mplstyle')

cfg_data = pd.read_csv(f"{ROOT}/data/ControlRoyG11042023.csv").rename(columns={"ID": "id"})
flanker_labels = ["median interference", "mean interference"]
nback_labels = ["d_prime", "# correct"]

for data_name, y_labels in zip(("flanker", "nback"), (flanker_labels, nback_labels)):
    data = pd.read_csv(f"{ROOT}/data/{data_name}_data.csv").merge(cfg_data, on="id")
    print(f"{data_name} N = {len(data.id.unique())}")
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(12, 5))
    for ax, (y_label, y) in zip(axs.flat, data[y_labels].items()):
        data_dict = {"Self-avoidance": data["self avoidance"], y_label: y}
        scatter(data_dict, ax, spearman_label=True)
    plt.show()
