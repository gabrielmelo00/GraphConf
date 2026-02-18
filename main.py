"""
CLI helper to process computed metrics

Example usage
```bash
# Print metrics as a LaTeX table row
python main.py metrics metrics.pkl print

# Generate a pdf coverage histogram
python main.py metrics metrics.pkl plot histogram --out histogram.pdf

# Generate a candidate vs conformal set size scatter plot
python main.py metrics metrics.pkl plot scatter --out scatter.pdf
```
"""

import argparse

from conformal import typst_plot
from conformal.metrics import Metrics


def make_args():
    parser = argparse.ArgumentParser(description="CLI helper")
    subparsers = parser.add_subparsers(dest="command", help="Primary command")

    #######################################################
    #                        METRICS                      #
    #######################################################

    metrics_parser = subparsers.add_parser(
        "metrics", help="Manipulate .pkl Metrics files"
    )
    metrics_parser.add_argument("path", type=str, help="Path to the pickle file")
    metrics_subparsers = metrics_parser.add_subparsers(
        dest="subcommand", help="Metrics actions"
    )

    ###################################
    #          PRINT (Latex row)      #
    ###################################

    metrics_subparsers.add_parser("print", help="Print metrics as LateX table row")

    ###################################
    #           PLOT (typst)          #
    ###################################

    plot_parser = metrics_subparsers.add_parser(
        "plot", help="Generate pdf plot with typst"
    )
    plot_parser.add_argument(
        "type", choices=["histogram", "scatter"], help="Type of plot to generate"
    )
    plot_parser.add_argument("--out", type=str, help="Path to save the output file")

    return parser.parse_args()


def main():

    args = make_args()

    if args.command == "metrics":
        metrics = Metrics.load(args.path)
        print(f"Loaded metrics from {args.path}")

        match args.subcommand:
            case "print":
                metrics.latex_metrics_row()
            case "plot":
                match args.type:
                    case "histogram":
                        out = args.out if args.out else "histogram.pdf"
                        typst_plot.histogram(metrics, out)

                    case "scatter":
                        out = args.out if args.out else "scatter.pdf"
                        typst_plot.scatter(metrics, out)


if __name__ == "__main__":
    main()
