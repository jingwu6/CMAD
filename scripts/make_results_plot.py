"""Plot accuracy vs SFT iteration for CMAD vs CMAD w/o diversity selection.

Usage: python3 scripts/make_results_plot.py --results-dir results/ --out plot.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory containing cmad_iterN.json and cmad_no_diversity_iterN.json")
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--out", default="results_plot.png")
    args = parser.parse_args()

    def load_accuracy(variant, iteration):
        key = f"{variant}_iter{iteration}"
        path = os.path.join(args.results_dir, f"{key}.json")
        with open(path) as f:
            data = json.load(f)
        return data[key]["accuracy"]

    iterations = list(range(1, args.num_iterations + 1))
    cmad_acc = [load_accuracy("cmad", i) for i in iterations]
    no_div_acc = [load_accuracy("cmad_no_diversity", i) for i in iterations]

    plt.figure(figsize=(6, 4))
    plt.plot(iterations, cmad_acc, marker="o", label="CMAD (with diversity selection)")
    plt.plot(iterations, no_div_acc, marker="s", linestyle="--", label="CMAD w/o diversity selection")
    plt.xlabel("SFT iteration")
    plt.ylabel("Accuracy")
    plt.title("Diversity selection prevents mental-set collapse")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"[cmad] make_results_plot wrote {args.out}: cmad={cmad_acc} no_diversity={no_div_acc}")


if __name__ == "__main__":
    main()
