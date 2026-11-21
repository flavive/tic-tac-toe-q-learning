import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_training(history_path: str, output_path: str) -> None:
    history = pd.read_csv(history_path)

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))

    left.plot(history["episode"], history["win_rate"], label="Win", linewidth=2)
    left.plot(history["episode"], history["draw_rate"], label="Draw", linewidth=2)
    left.plot(history["episode"], history["loss_rate"], label="Loss", linewidth=2)
    left.set_title("Agent performance")
    left.set_xlabel("Training episode")
    left.set_ylabel("Rate in greedy evaluation")
    left.set_ylim(-0.02, 1.02)
    left.legend()
    left.grid(alpha=0.25)

    right.plot(history["episode"], history["epsilon"], color="purple", label="Epsilon")
    right.plot(history["episode"], history["states_learned"], color="orange", label="Learned states")
    right.set_title("Exploration and learning")
    right.set_xlabel("Training episode")
    right.legend()
    right.grid(alpha=0.25)

    fig.suptitle("Q-learning Tic-Tac-Toe training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="results/training_history.csv")
    parser.add_argument("--output", default="results/training_plot.png")
    args = parser.parse_args()
    plot_training(args.history, args.output)


if __name__ == "__main__":
    main()
