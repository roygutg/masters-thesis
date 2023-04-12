import pandas as pd
from utils import ROOT

flanker_df = pd.read_csv(f"{ROOT}/data/all_flanker_data.csv")
flanker_df[["mean interference", "median interference"]] = flanker_df.groupby("id")[["mean_rt", "median_rt"]].diff()
interference_df = flanker_df[["id", "mean interference", "median interference"]].dropna().reset_index(drop=True)
interference_df.to_csv(f"{ROOT}/data/flanker_data.csv", index=False)
