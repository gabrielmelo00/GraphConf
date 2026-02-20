import json
import pickle
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class Metrics:
    """Conformal prediction metrics with export utilities"""

    def __init__(
        self,
        correct_coverage: list[bool],
        candidate_sizes: list[int],
        conformal_sizes: list[int],
    ):
        self.correct_coverage = np.array(correct_coverage, dtype=bool)
        self.candidate_sizes = np.array(candidate_sizes)
        self.conformal_sizes = np.array(conformal_sizes)

    def save(self, path: str):
        """Save to a pickled file"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load from a pickled file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def coverage(self) -> float:
        return self.correct_coverage.mean()

    @property
    def mean_set_size(self) -> float:
        return self.conformal_sizes[self.correct_coverage].mean()

    @property
    def median_set_size(self) -> float:
        return np.median(self.conformal_sizes[self.correct_coverage])

    @property
    def mean_candidate_size(self) -> float:
        return self.candidate_sizes[self.correct_coverage].mean()

    @property
    def median_candidate_size(self) -> float:
        return np.median(self.candidate_sizes[self.correct_coverage])

    @property
    def mean_reduction(self) -> float:
        correct_conformal = self.conformal_sizes[self.correct_coverage]
        correct_candidate = self.candidate_sizes[self.correct_coverage]
        return 1 - (correct_conformal / correct_candidate).mean()

    @property
    def median_reduction(self) -> float:
        correct_conformal = self.conformal_sizes[self.correct_coverage]
        correct_candidate = self.candidate_sizes[self.correct_coverage]
        return 1 - np.median(correct_conformal / correct_candidate)

    @property
    def empty_rate(self) -> float:
        return (self.conformal_sizes == 0).mean()

    def __len__(self):
        return len(self.correct_coverage)

    ###########################################################################
    #                             Export utilities                            #
    ###########################################################################

    def json_histogram(self, path: str, target=90, bin_size=50):
        """Save coverage rate by candidate set size in a json file"""

        n_bins = int(np.ceil(self.candidate_sizes.max() / bin_size))
        correct = np.zeros(n_bins)
        total = np.zeros(n_bins)

        for i in range(len(self)):
            bin = self.candidate_sizes[i] // bin_size
            total[bin] += 1
            if self.correct_coverage[i]:
                correct[bin] += 1

        total[total == 0] = 1  # Avoid division by zero for empty bins

        data = {"target": target, "bin_size": bin_size, "bins": list(correct / total)}

        json.dump(data, open(path, "w"), indent=4)

    def csv_set_sizes(self, path: str):
        """Save the set sizes to a csv file for later plotting"""

        df = pl.DataFrame(
            {
                "candidate_size": self.candidate_sizes,
                "conformal_size": self.conformal_sizes,
                "in_conformal": self.correct_coverage,
            }
        )
        df.write_csv(path)

    def latex_metrics_row(self):
        """Print LateX code for a row of the metrics table"""

        strings: list[str] = [
            f"${100 * self.coverage:.1f}\%$",
            f"${round(self.mean_set_size)}$",
            f"${round(self.median_set_size)}$",
            f"${100 * self.mean_reduction:.1f}\%$",
            f"${100 * self.median_reduction:.1f}\%$",
            f"${100 * self.empty_rate:.1f}\%$",
        ]

        print(" & ".join(strings))

    def typst_metrics_row(self, method: str):
        """Print the typst code for a row of the metrics table"""

        strings: list[str] = [f"[{method}]"]

        strings.append(f"${100 * self.coverage:.1f}%$")
        strings.append(f"${round(self.mean_set_size)}$")
        strings.append(f"${round(self.median_set_size)}$")
        strings.append(f"${100 * self.mean_reduction:.1f}%$")
        strings.append(f"${100 * self.median_reduction:.1f}%$")
        strings.append(f"${100 * self.empty_rate:.1f}%$")

        print(", ".join(strings))

    ###########################################################################
    #                            Plotting utilities                           #
    ###########################################################################

    def plot_set_sizes(self):
        """Compare conformal set sizes vs candidate set sizes"""

        n_bins = int(np.ceil(self.candidate_sizes.max() / 50))
        correct = np.zeros(n_bins)
        total = np.zeros(n_bins)

        for i in range(len(self)):
            bin = self.candidate_sizes[i] // 50
            total[bin] += 1
            if self.correct_coverage[i]:
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

    def plot_binned_coverage_rate(self):
        """Plot coverage rate binned by candidate set size"""

        correct_conf = self.conformal_sizes[self.correct_coverage]
        correct_cand = self.candidate_sizes[self.correct_coverage]
        plt.scatter(
            correct_cand, correct_conf, alpha=0.5, color="blue", label="GT in conformal"
        )

        wrong_conf = self.conformal_sizes[~self.correct_coverage]
        wrong_cand = self.candidate_sizes[~self.correct_coverage]
        plt.scatter(
            wrong_cand, wrong_conf, alpha=0.5, color="red", label="GT not in conformal"
        )

        plt.plot(
            [0, self.candidate_sizes.max()],
            [0, self.candidate_sizes.max()],
            "k--",
        )

        plt.grid()
        plt.xlabel("Candidate Set Size")
        plt.ylabel("Conformal Set Size")
        plt.legend()
        plt.title("Conformal Set Size vs Candidate Set Size")
