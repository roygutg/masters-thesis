import numpy as np
import pandas as pd
from simCFG import CreativeForagingAgent
from ComDePy.numerical import MultiAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils import scatter, ROOT

plt.style.use('comdepri.mplstyle')


class SimulationException(Exception):
    pass


class SelfAvoidingAgent(CreativeForagingAgent):
    def __init__(self, avoidance_span: int, *agent_params, **agent_kw):
        if avoidance_span <= 0:
            raise SimulationException("avoidance span must be positive")

        self.avoidance_span = avoidance_span
        super().__init__(*agent_params, **agent_kw)

    def _sim(self):
        """
        Runs the sim. Fails if already ran.
        """
        for t in range(1, self.max_steps):
            memory_start = max(0, t - self.avoidance_span)
            self.a[t], self.b[t], self.s[t] = self._step(self.a[t - 1], self.b[t - 1], self.s[memory_start:t])

        self._segment_explore_exploit()


class Simulator:
    def __init__(self, n_agents, avoidance_spans, log_g, log_alpha):
        self.n_agents = n_agents
        self.avoidance_spans = avoidance_spans
        self.log_g = log_g
        self.log_alpha = log_alpha

        self.summed_df = None
        self.raw_df = None
        self.ran = False

    def get_sim_results(self):
        if self.ran:
            raise SimulationException("already ran")

        data_dir = rf"{ROOT}\data\avoidance_simulations"
        sim_param_string = f"n={self.n_agents}_log_g={self.log_g}_log_alpha={self.log_alpha}"
        raw_filename = fr"{data_dir}\raw_{sim_param_string}.csv"
        summed_filename = fr"{data_dir}\summed_{sim_param_string}.csv"
        if os.path.isfile(raw_filename) and os.path.isfile(summed_filename):
            self.raw_df = pd.read_csv(raw_filename)
            self.summed_df = pd.read_csv(summed_filename)
        else:
            self.raw_df, self.summed_df = self._simulate()
            self.raw_df.to_csv(raw_filename, index=False)
            self.summed_df.to_csv(summed_filename, index=False)

        self.ran = True

    def _simulate(self):
        def gather(agent):
            explore_durations, exploit_durations = agent.get_phase_durations()
            return (agent.get_self_avoidance(), agent.get_n_clusters(), agent.converged(), np.median(explore_durations),
                    np.median(exploit_durations), np.std(explore_durations), np.std(exploit_durations))

        measure_names = ("self-avoidance", "# clusters", "converged", "median explore",
                         "median exploit", "sd explore", "sd exploit")
        raw_data = np.array([]).reshape((0, len(measure_names)))
        summed_data = {"Avoidance span": np.repeat(self.avoidance_spans, len(measure_names)),
                       "measure_name": measure_names * len(self.avoidance_spans),
                       "ci_low": [], "ci_high": []}

        for avoidance_span in tqdm(self.avoidance_spans, desc="avoidance spans"):
            ma = MultiAgent(self.n_agents,
                            lambda: SelfAvoidingAgent(avoidance_span, 10 ** self.log_g, 10 ** self.log_alpha),
                            gather)
            ma.sim()
            raw_data = np.vstack((raw_data, ma.get_data()))
            ci_low, ci_high = ma.bootstrap_ci(vectorized=True, axis=0)
            summed_data["ci_low"].extend(ci_low)
            summed_data["ci_high"].extend(ci_high)

        raw_df = pd.DataFrame(dict(zip(measure_names, raw_data.T)))
        raw_df["Avoidance span"] = np.repeat(avoidance_spans, n_agents)
        summed_df = pd.DataFrame(summed_data)
        summed_df["mean"] = (summed_df["ci_low"] + summed_df["ci_high"]) / 2
        summed_df["half_ci_width"] = (summed_df["ci_low"] - summed_df["ci_high"]) / 2

        return raw_df, summed_df

    def plot_all(self):
        if not self.ran:
            raise SimulationException("must run first")

        for measure_name, measure_df in self.summed_df.groupby("measure_name"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"{measure_name}\n" + rf"$log_{{10}}g={self.log_g}$, $log_{{10}}\alpha={self.log_alpha}$")

            scatter(measure_df[["Avoidance span", "mean"]], yerr=measure_df["half_ci_width"],
                    spearman_label=True, ax=ax1)
            if measure_name != "self-avoidance":
                raw_vars_to_plot = ["self-avoidance", measure_name]
            else:
                raw_vars_to_plot = ["Avoidance span", "self-avoidance"]
            scatter(self.raw_df[raw_vars_to_plot], spearman_label=True, ax=ax2, alpha=0.2)
            plt.show()

    def plot_for_paper(self):
        if not self.ran:
            raise SimulationException("must run first")

        y_labels = ("Mean self-avoidance",
                    "Fraction of agent with\nexplore-exploit behavior",
                    "Averaged median\nexploration steps")
        measure_names = ("self-avoidance", "converged", "median explore")
        fig, axs = plt.subplots(1, len(measure_names), figsize=(2 + 5 * len(measure_names), 5), sharex=True)

        for ax, measure_name, y_label in zip(axs, measure_names, y_labels):
            measure_df = self.summed_df[self.summed_df["measure_name"] == measure_name]
            scatter(measure_df[["Avoidance span", "mean"]], yerr=measure_df["half_ci_width"], ax=ax)
            ax.set_ylabel(y_label)
        plt.savefig(f"{ROOT}/figs/avoidance_span.pdf", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    n_agents = 1000
    avoidance_spans = range(1, 21)
    log_g = 0
    log_alpha = 1

    s = Simulator(n_agents, avoidance_spans, log_g, log_alpha)
    s.get_sim_results()
    s.plot_all()
    # s.plot_for_paper()
