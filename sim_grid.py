import numpy as np
import pandas as pd
from simCFG import CreativeForagingAgent
from ComDePy.numerical import MultiAgentGrid
import matplotlib.pyplot as plt
import os
from utils import scatter, ROOT
from scipy.stats import bootstrap

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
    def __init__(self, n_agents, avoidance_spans, log_gs, log_alphas):
        self.n_agents = n_agents
        self.avoidance_spans = np.array(avoidance_spans)
        self.log_gs = np.array(log_gs)
        self.log_alphas = np.array(log_alphas)

        self.summed_df = None
        self.ran = False

    def get_sim_results(self):
        if self.ran:
            raise SimulationException("already ran")

        data_dir = rf"{ROOT}\data\avoidance_simulations"
        sim_param_string = f"n={self.n_agents}_log_gs={self.log_gs}_log_alphas={self.log_alphas}"
        summed_filename = fr"{data_dir}\summed_{sim_param_string}.csv"
        if os.path.isfile(summed_filename):
            self.summed_df = pd.read_csv(summed_filename)
        else:
            self.summed_df = self._simulate()
            self.summed_df.to_csv(summed_filename, index=False)

        self.ran = True

    def _simulate(self):
        def gather(agent):
            explore_durations, exploit_durations = agent.get_phase_durations()
            return (agent.get_self_avoidance(), agent.get_n_clusters(), agent.converged(), np.median(explore_durations),
                    np.median(exploit_durations), np.std(explore_durations), np.std(exploit_durations))

        mag = MultiAgentGrid(n_agents,
                             [avoidance_spans, 10 ** log_gs, 10 ** log_alphas],
                             SelfAvoidingAgent,
                             gather)
        mag.sim()
        mag_data = mag.get_results_grid()

        measure_names = ("self-avoidance", "# clusters", "converged", "median explore",
                         "median exploit", "sd explore", "sd exploit")
        summed_dict = {"avoidance_span": [], "measure_name": [], "mean": [], "ci_half_width": []}
        for i, measure_name in zip(range(mag_data.shape[-1]), measure_names):
            measure_data = np.mean(mag_data[:, :, :, :, i], axis=(1, 2))
            ci_low, ci_high = bootstrap((measure_data,), np.nanmean, vectorized=True, axis=1).confidence_interval
            is_nan = np.isnan((ci_low, ci_high))
            if is_nan.any():
                statistic_of_data = np.nanmean(measure_data, axis=1)
                arr = np.array((ci_low, ci_high))
                arr[is_nan] = np.array((statistic_of_data, statistic_of_data))[is_nan]
                ci_low, ci_high = arr

            summed_dict["measure_name"] += [measure_name] * len(self.avoidance_spans)
            summed_dict["avoidance_span"].extend(self.avoidance_spans)
            summed_dict["mean"].extend((ci_high + ci_low) / 2)
            summed_dict["ci_half_width"].extend((ci_high - ci_low) / 2)

        return pd.DataFrame(summed_dict)

    def plot_all(self):
        if not self.ran:
            raise SimulationException("must run first")

        for measure_name, measure_df in self.summed_df.groupby("measure_name"):
            plt.figure(figsize=(8, 5))
            plt.suptitle(measure_name)
            log_g_linspace_params = f"{min(self.log_gs)},{max(self.log_gs)},{len(self.log_gs)}"
            log_alpha_linspace_params = f"{min(self.log_alphas)},{max(self.log_alphas)},{len(self.log_alphas)}"
            plt.title(rf"$log_{{10}}g=$linspace({log_g_linspace_params})" + "\n" +
                      rf"$log_{{10}}\alpha=$linspace({log_alpha_linspace_params})")

            scatter(measure_df[["avoidance_span", "mean"]], yerr=measure_df["ci_half_width"], spearman_label=True)
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
    n_agents = 100
    avoidance_spans = range(1, 7)  # change to up to 21
    log_gs = np.linspace(0, 2, 6)  # change to len 15
    log_alphas = np.linspace(0, 2, 6)  # change to len 15

    s = Simulator(n_agents, avoidance_spans, log_gs, log_alphas)
    s.get_sim_results()
    s.plot_all()
    # s.plot_for_paper()
