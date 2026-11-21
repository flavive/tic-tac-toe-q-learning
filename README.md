# Tic-Tac-Toe Q-Learning Agent

A reproducible educational project that trains a tabular Q-learning agent to play Tic-Tac-Toe. The agent starts with a random policy and gradually learns which moves lead to wins or draws against a random opponent.

## Features

- Reproducible Q-learning implementation
- Random opponent for clear baseline behavior
- Periodic greedy evaluation (separate from training epsilon)
- CSV and JSON training history
- Q-table snapshots at early, halfway, and final checkpoints
- Report-ready PNG performance chart
- Interactive command-line play against the trained agent
- Unit tests

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/tictactoe.py --episodes 50000 --seed 42
python src/plot_results.py
python src/play.py
pytest
```

On Windows, use `py` instead of `python` if needed.

## Recorded outputs

After training, `results/` contains:

| File | Purpose |
|---|---|
| `training_history.csv` | Episode, epsilon, win/draw/loss rates, and learned-state count |
| `training_history.json` | The same evaluation history in JSON |
| `q_table_snapshots.csv` | Q-values recorded at early, middle, and final checkpoints |
| `agent.json` | Final learned Q-table and agent settings |
| `training_plot.png` | Report-ready training chart |

CSV outputs can be inserted directly into reports or converted into a web dashboard. Commit the `results/` files so the experiment can be compared across runs.

## How it works

The agent uses epsilon-greedy action selection. It explores random empty squares when epsilon is high, and increasingly chooses the highest-Q-value empty square as epsilon decays.

Rewards:

- `+1` for a win
- `+0.2` for a draw
- `-1` for a loss
- `0` for a continuing move

The Bellman update is:

`Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))`

Tic-Tac-Toe is small enough for a tabular Q-table: every board state maps to the Q-value of each legal action.

## Reproducibility

The random seeds control environment and policy randomness. Use the same episode count and seed to reproduce the agent. The implementation also records epsilon and learned-state counts at evaluation checkpoints.

## Project structure

```text
src/
  tictactoe.py      # environment and training loop
  plot_results.py   # report chart
  play.py           # human-vs-agent CLI
results/            # generated artifacts (created by training)
tests/              # unit tests
```

## Suggested report results

Include:

1. A line chart of win/draw/loss rates versus training episode.
2. A chart of epsilon and states learned versus episode.
3. Example board states and chosen actions at the 500-, 25,000-, and 50,000-episode checkpoints.
4. A replay of one early random game and one final greedy game.
5. Final model settings and seed.
