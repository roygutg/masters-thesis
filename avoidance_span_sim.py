import numpy as np
import pandas as pd
from simCFG import CreativeForagingAgent
from ComDePy.numerical import MultiAgentGrid
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import os
from scipy.stats import spearmanr, bootstrap
from utils import ROOT

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


def _add_spearman_label(x, y, ax):
    rho, p = spearmanr(x, y)
    if np.isfinite(rho * p):
        rho_label = fr"$\rho={round(rho, 2)}$"
        p_round_order = 3
        if p == 0:
            p_label = "$p=0$"
        elif p >= 10 ** -p_round_order:
            p_label = f"$p={round(p, p_round_order)}$"
        else:
            p_label = f"$p<10^{{{int(np.log10(p))}}}$"

        # placement solution from https://stackoverflow.com/a/59109053:
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="w", ec="w", lw=0, alpha=0)] * 2
        labels = [rho_label, p_label]
        ax.legend(handles, labels, title="Spearman", handlelength=0, handletextpad=0,
                  loc="best", fontsize="small", fancybox=True, framealpha=0.7)


class Simulator:
    def __init__(self, n_agents, avoidance_spans, log_gs, log_alphas, gather, measure_names):
        self.n_agents = n_agents
        self.avoidance_spans = avoidance_spans
        self.log_gs = log_gs
        self.log_alphas = log_alphas
        self.gather = gather
        self.measure_names = measure_names

        self.raw_results = None
        self.summed_df = None
        self.ran = False

    def get_sim_results(self):
        if self.ran:
            raise SimulationException("already ran")

        sim_param_string = fr"n={self.n_agents}_log_g={min(self.log_gs), max(self.log_gs)}_log_alpha=" \
                           f"{min(self.log_alphas), max(self.log_alphas)}"

        raw_filename = fr"{ROOT}\data\raw_{sim_param_string}.npy"
        if os.path.isfile(raw_filename):
            self.raw_results = np.load(raw_filename)
        else:
            self._simulate()
            np.save(raw_filename, self.raw_results)

        summed_filename = fr"{ROOT}\data\summed_{sim_param_string}.csv"
        if os.path.isfile(summed_filename):
            self.summed_df = pd.read_csv(summed_filename)
        else:
            self._get_summed_results()
            self.summed_df.to_csv(summed_filename, index=False)

        self.ran = True

    def _simulate(self):
        mag = MultiAgentGrid(self.n_agents,
                             [self.avoidance_spans, 10 ** self.log_gs, 10 ** self.log_alphas],
                             SelfAvoidingAgent,
                             self.gather)
        mag.sim()
        self.raw_results = mag.get_results_grid()

    def _get_summed_results(self):
        if self.raw_results is None:
            raise SimulationException("Can't summarize results, raw results missing")
        if self.summed_df is not None:
            raise SimulationException("Can't summarize results, summed_df already exists")

        self.summed_df = pd.DataFrame({"Avoidance span": np.repeat(self.avoidance_spans, len(self.measure_names)),
                                       "measure_name": self.measure_names * len(self.avoidance_spans)})

        data = np.nanmean(self.raw_results, axis=3)  # average over agents
        data = data.reshape((len(self.avoidance_spans), -1, len(self.measure_names)))  # flatten g, alpha
        ci_lows, ci_highs = bootstrap((data,), np.nanmean, vectorized=True, axis=1).confidence_interval
        self.summed_df["mean"] = np.nanmean(data, axis=1).reshape(-1)
        self.summed_df["error_minus"] = self.summed_df["mean"] - ci_lows.reshape(-1)
        self.summed_df["error_plus"] = ci_highs.reshape(-1) - self.summed_df["mean"]

    def plot_results(self):
        if not self.ran:
            raise SimulationException("must run first")

        y_labels = ["Mean self-avoidance", "Fraction of agents with\nexplore-exploit behavior"]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for i, (ax, y_label, converged_only) in enumerate(zip(axs, y_labels, [True, False])):
            data = self.raw_results.copy()
            if converged_only:
                idx_not_converged = self.raw_results[:, :, :, :, 1] == 0
                data[idx_not_converged] = np.nan
            local_means = np.nanmean(data, axis=3)  # average over agents
            local_means = local_means.reshape(
                (len(self.avoidance_spans), -1, len(self.measure_names)))  # flatten g, alpha
            local_measure_means = local_means[:, :, i]
            overall_measure_means = np.mean(local_measure_means, axis=1)

            # local means:
            legend_flag = True
            for avoidance_span, ys in zip(self.avoidance_spans, local_measure_means):
                label = None
                if legend_flag:
                    label = r"Local $\left(g, \alpha\right)$"
                    legend_flag = False

                ax.scatter([avoidance_span] * len(ys), ys, color="gray", alpha=0.2, label=label)

            # overall means:
            ax.scatter(self.avoidance_spans, overall_measure_means, edgecolors="k", s=100, label="Overall")
            print(y_label, spearmanr(self.avoidance_spans, overall_measure_means))

            ax.set_xlabel("Avoidance span")
            ax.set_ylabel(y_label)
            ax.legend(fontsize=12)

        plt.savefig(fr"{ROOT}\figs\avoidance_span.pdf", bbox_inches="tight")
        plt.show()


def gather(agent):
    return agent.get_self_avoidance(), agent.converged()


if __name__ == '__main__':
    avoidance_spans = range(1, 11)
    log_gs = np.linspace(-0.2, 0.2, 15)
    log_alphas = np.linspace(0.8, 1.2, 15)
    n_agents = 1000
    measure_names = ("self-avoidance", "converged")

    s = Simulator(n_agents, avoidance_spans, log_gs, log_alphas, gather, measure_names)
    s.get_sim_results()
    s.plot_results()
