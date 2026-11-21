import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


EMPTY = 0
AGENT = 1
OPPONENT = -1
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def state_key(board: np.ndarray) -> tuple[int, ...]:
    """Canonical state: rotate/reflection symmetry is not required for learning."""
    return tuple(int(value) for value in board.flatten())


def winner(board: np.ndarray) -> int:
    for a, b, c in WIN_LINES:
        value = int(board[a // 3, a % 3] + board[b // 3, b % 3] + board[c // 3, c % 3])
        if value == 3:
            return AGENT
        if value == -3:
            return OPPONENT
    return EMPTY


def is_full(board: np.ndarray) -> bool:
    return bool(np.all(board != EMPTY))


def available_actions(board: np.ndarray) -> list[int]:
    return [index for index, value in enumerate(board.flatten()) if value == EMPTY]


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.35,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.9995,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q_values: dict[tuple[int, ...], dict[int, float]] = defaultdict(dict)

    def q(self, board: np.ndarray, action: int) -> float:
        return self.q_values[state_key(board)].get(action, 0.0)

    def choose_action(self, board: np.ndarray, training: bool = True) -> int:
        actions = available_actions(board)
        if not actions:
            raise ValueError("No available actions")
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        scores = [self.q(board, action) for action in actions]
        maximum = max(scores)
        best = [action for action, score in zip(actions, scores) if abs(score - maximum) < 1e-12]
        return self.rng.choice(best)

    def update(self, board: np.ndarray, action: int, reward: float, next_board: np.ndarray) -> None:
        state = state_key(board)
        old_value = self.q_values[state].get(action, 0.0)
        next_actions = available_actions(next_board)
        next_max = max((self.q(next_board, next_action) for next_action in next_actions), default=0.0)
        target = reward if not next_actions else reward + self.gamma * next_max
        self.q_values[state][action] = old_value + self.alpha * (target - old_value)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            state: {str(action): value for action, value in actions.items()}
            for state, actions in self.q_values.items()
        }
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_values": serializable,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "QLearningAgent":
        data = json.loads(path.read_text(encoding="utf-8"))
        agent = cls(
            alpha=data["alpha"], gamma=data["gamma"], epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"], epsilon_decay=data["epsilon_decay"],
        )
        for state_text, actions in data["q_values"].items():
            state = tuple(int(value) for value in state_text.strip("()[ ]").replace(",", " ").split())
            agent.q_values[state] = {int(action): value for action, value in actions.items()}
        return agent


def make_move(board: np.ndarray, action: int, player: int) -> np.ndarray:
    next_board = board.copy()
    if next_board.flatten()[action] != EMPTY:
        raise ValueError("Cannot move in an occupied square")
    next_board[action // 3, action % 3] = player
    return next_board


def play_training_episode(agent: QLearningAgent, rng: random.Random) -> str:
    board = np.zeros((3, 3), dtype=int)

    while True:
        state = board.copy()
        action = agent.choose_action(board, training=True)
        board = make_move(board, action, AGENT)

        result = winner(board)
        if result == AGENT:
            agent.update(state, action, 1.0, board)
            return "win"
        if is_full(board):
            agent.update(state, action, 0.2, board)
            return "draw"

        opponent_action = rng.choice(available_actions(board))
        board = make_move(board, opponent_action, OPPONENT)

        result = winner(board)
        if result == OPPONENT:
            agent.update(state, action, -1.0, board)
            return "loss"
        if is_full(board):
            agent.update(state, action, 0.2, board)
            return "draw"

        agent.update(state, action, 0.0, board)


def evaluate(agent: QLearningAgent, episodes: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    counts = {"win": 0, "draw": 0, "loss": 0}

    for _ in range(episodes):
        board = np.zeros((3, 3), dtype=int)
        while True:
            action = agent.choose_action(board, training=False)
            board = make_move(board, action, AGENT)
            if winner(board) == AGENT:
                counts["win"] += 1
                break
            if is_full(board):
                counts["draw"] += 1
                break

            action = rng.choice(available_actions(board))
            board = make_move(board, action, OPPONENT)
            if winner(board) == OPPONENT:
                counts["loss"] += 1
                break
            if is_full(board):
                counts["draw"] += 1
                break

    total = float(episodes)
    return {name: count / total for name, count in counts.items()}


def write_q_table_snapshot(agent: QLearningAgent, output_path: Path, episode: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for state, actions in sorted(agent.q_values.items(), key=lambda item: str(item[0])):
        board = np.array(state).reshape(3, 3)
        for action, value in sorted(actions.items(), key=lambda item: item[0]):
            rows.append({
                "episode": episode,
                "board": "".join({-1: "O", 0: "-", 1: "X"}[int(cell)] for cell in state),
                "action": action,
                "q_value": value,
                "q_source": "learned",
            })

    with output_path.open("a", encoding="utf-8") as handle:
        if episode == 0:
            handle.write("episode,board,action,q_value,q_source\n")
        for row in rows:
            handle.write(f'{row["episode"]},{row["board"]},{row["action"]},{row["q_value"]:.12f},{row["q_source"]}\n')


def train(
    episodes: int = 50_000,
    seed: int = 42,
    evaluation_every: int = 500,
    evaluation_games: int = 500,
    output_dir: Path = Path("results"),
) -> tuple[QLearningAgent, list[dict]]:
    agent = QLearningAgent(seed=seed)
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_csv = output_dir / "training_history.csv"
    snapshot_csv = output_dir / "q_table_snapshots.csv"
    history_csv.write_text("episode,epsilon,win_rate,draw_rate,loss_rate,states_learned\n", encoding="utf-8")
    snapshot_csv.write_text("episode,board,action,q_value,q_source\n", encoding="utf-8")

    history = []
    window = {"win": 0, "draw": 0, "loss": 0}

    for episode in range(1, episodes + 1):
        result = play_training_episode(agent, rng)
        window[result] += 1
        agent.decay_epsilon()

        if episode % evaluation_every == 0:
            rates = evaluate(agent, evaluation_games, seed + episode)
            record = {
                "episode": episode,
                "epsilon": agent.epsilon,
                "win_rate": rates["win"],
                "draw_rate": rates["draw"],
                "loss_rate": rates["loss"],
                "states_learned": len(agent.q_values),
            }
            history.append(record)
            with history_csv.open("a", encoding="utf-8") as handle:
                handle.write(
                    f'{episode},{agent.epsilon:.6f},{record["win_rate"]:.6f},'
                    f'{record["draw_rate"]:.6f},{record["loss_rate"]:.6f},{len(agent.q_values)}\n'
                )

            if episode in (evaluation_every, episodes // 2, episodes):
                write_q_table_snapshot(agent, snapshot_csv, episode)

    agent.save(output_dir / "agent.json")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training complete: {episodes:,} episodes")
    print(f"Final win/draw/loss: {history[-1]['win_rate']:.1%} / {history[-1]['draw_rate']:.1%} / {history[-1]['loss_rate']:.1%}")
    print(f"Learned states: {len(agent.q_values):,}")
    return agent, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Q-learning Tic-Tac-Toe agent")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-every", type=int, default=500)
    parser.add_argument("--evaluation-games", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    train(args.episodes, args.seed, args.evaluation_every, args.evaluation_games, args.output_dir)


if __name__ == "__main__":
    main()
