import argparse
import random

import numpy as np

from tictactoe import AGENT, OPPONENT, QLearningAgent, available_actions, is_full, make_move, winner


def render(board: np.ndarray) -> str:
    symbols = {0: " ", 1: "X", -1: "O"}
    rows = [" | ".join(symbols[int(value)] for value in row) for row in board]
    return "\n---------\n".join(rows)


def replay(agent: QLearningAgent, seed: int = 42) -> None:
    rng = random.Random(seed)
    board = np.zeros((3, 3), dtype=int)
    print("You are O. Enter a square number 1–9.")

    while True:
        print("\n" + render(board))
        action = agent.choose_action(board, training=False)
        board = make_move(board, action, AGENT)
        print("\nAgent moves:")
        print(render(board))

        if winner(board) == AGENT:
            print("\nAgent wins.")
            return
        if is_full(board):
            print("\nDraw.")
            return

        legal = available_actions(board)
        while True:
            try:
                choice = int(input("Choose 1–9: ")) - 1
            except ValueError:
                continue
            if choice in legal:
                break
            print("Choose an empty square.")

        board = make_move(board, choice, OPPONENT)
        if winner(board) == OPPONENT:
            print("\n" + render(board))
            print("You win.")
            return
        if is_full(board):
            print("\n" + render(board))
            print("Draw.")
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/agent.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    replay(QLearningAgent.load(__import__("pathlib").Path(args.model)), args.seed)


if __name__ == "__main__":
    main()
