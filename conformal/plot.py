"""Plotting utilities to reproduce the same figures as in the paper."""

import matplotlib.pyplot as plt
import numpy as np

from conformal.calibrate import Metrics


def plot_set_sizes(metrics: Metrics):
    """Compare conformal set sizes vs candidate set sizes"""

    n_bins = int(np.ceil(metrics.candidate_sizes.max() / 50))
    correct = np.zeros(n_bins)
    total = np.zeros(n_bins)

    for i in range(len(metrics)):
        bin = metrics.candidate_sizes[i] // 50
        total[bin] += 1
        if metrics.correct_coverage[i]:
            correct[bin] += 1

    plt.bar(
        np.arange(n_bins) * 50,
        correct / total,
        width=40,
    )

    plt.plot([0, n_bins * 50], [0.9, 0.9], "r--", label="Target: 90%")

    plt.xlabel("Candidate Set Size (binned)")
    plt.ylabel("Coverage Rate")
    plt.grid()
    plt.legend()
    plt.title("Coverage Rate by Candidate Set Size")


def plot_binned_coverage_rate(metrics: Metrics):
    """Plot coverage rate binned by candidate set size"""

    correct_conf = metrics.conformal_sizes[metrics.correct_coverage]
    correct_cand = metrics.candidate_sizes[metrics.correct_coverage]
    plt.scatter(
        correct_cand, correct_conf, alpha=0.5, color="blue", label="GT in conformal"
    )

    wrong_conf = metrics.conformal_sizes[~metrics.correct_coverage]
    wrong_cand = metrics.candidate_sizes[~metrics.correct_coverage]
    plt.scatter(
        wrong_cand, wrong_conf, alpha=0.5, color="red", label="GT not in conformal"
    )

    plt.plot(
        [0, metrics.candidate_sizes.max()],
        [0, metrics.candidate_sizes.max()],
        "k--",
    )

    plt.grid()
    plt.xlabel("Candidate Set Size")
    plt.ylabel("Conformal Set Size")
    plt.legend()
    plt.title("Conformal Set Size vs Candidate Set Size")
