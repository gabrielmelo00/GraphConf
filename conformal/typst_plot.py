"""
Produce 2D plots from metrics with typst
Note that this uses a temporary directory temp/
"""

import os
import shutil

import typst

from conformal.metrics import Metrics


def histogram(metrics: Metrics, out: str):
    # Fails if it exists. We will put stuff in this directory, then erase it.
    os.mkdir("temp")

    metrics.json_histogram("temp/histogram.json")

    content = f"""{HEADER}

{HIST_TEMPLATE}

#coverage_rates("temp/histogram.json")
"""
    typst.compile(content.encode(), output=out)  # type: ignore
    print("Saved histogram to ", out)

    shutil.rmtree("temp")


def scatter(metrics: Metrics, out: str):
    # Fails if it exists. We will put stuff in this directory, then erase it.
    os.mkdir("temp")

    metrics.csv_set_sizes("temp/scatter.csv")
    cov = f"[{100 * metrics.coverage:.1f}%]"
    miss = f"[{100 * (1 - metrics.coverage):.1f}%]"

    content = f"""{HEADER}

{SCATTER_TEMPLATE}

#conformal_vs_candidate("temp/scatter.csv", {cov}, {miss})
"""

    typst.compile(content.encode(), output=out)  # type: ignore
    print("Saved scatter plot to ", out)

    shutil.rmtree("temp")


###########################################################
#                         TEMPLATES                       #
###########################################################

HEADER = """
#import "@preview/lilaq:0.5.0" as lq
#set page(height: auto, width: auto, margin: 0.2cm)
#let slice = (arr, step) => arr.chunks(step).map(array.first)
"""

HIST_TEMPLATE = """
#let coverage_rates = path => {
  let (target, bin_size, bins) = json(path)

  lq.diagram(
    title: [Coverage Rate by Candidate Set Size],
    xlabel: [Candidate Set Size (binned)],
    ylabel: [Coverage Rate],
    legend: (position: horizon + right),

    xaxis: (
      lim: (-bin_size + 1, bins.len() * bin_size - 1),
      subticks: none,
      auto-exponent-threshold: 10000000,
    ),

    lq.bar(
      range(bins.len()).map(x => x * bin_size),
      bins,
      width: bin_size * 0.8,
      stroke: black + 0.5pt,
    ),

    lq.line(
      (-bin_size, target / 100),
      (bins.len() * bin_size, target / 100),
      stroke: (dash: "densely-dashed", thickness: 1pt, paint: red),
      label: [Target: ] + str(target) + [%],
    ),
  )
}
"""

SCATTER_TEMPLATE = """
#let conformal_vs_candidate = (path, cov, miss, step: 10) => {
  let (candidate_size, conformal_size, in_conformal) = lq.load-txt(
    read(path),
    header: true,
    converters: (
      candidate_size: int,
      conformal_size: int,
      in_conformal: s => s == "true",
    ),
  )

  let candidate_size = slice(candidate_size, step)
  let conformal_size = slice(conformal_size, step)
  let in_conformal = slice(in_conformal, step)

  let max_conformal = calc.max(..conformal_size)
  let max_candidate = calc.max(..candidate_size)


  lq.diagram(
    title: [Conformal Set Size vs Candidate Set Size],
    xlabel: [Candidate Set Size],
    ylabel: [Conformal Set Size],
    legend: lq.legend(
      circle(fill: blue, radius: 1.5pt),
      text(size: 6pt)[GT in conformal (cov = #cov)],
      circle(fill: red, radius: 1.5pt),
      text(size: 6pt)[GT not in conformal (miss = #miss)],
      position: top + left,
    ),

    xaxis: (
      auto-exponent-threshold: 10000000,
    ),

    lq.scatter(
      candidate_size,
      conformal_size,
      color: in_conformal.map(in_c => if in_c { blue } else { red }),
      alpha: 50%,
      stroke: none,
    ),

    lq.line(
      (0, 0),
      (max_candidate, max_candidate),
      stroke: (dash: "densely-dashed", thickness: 1pt),
    ),
  )
}
"""
