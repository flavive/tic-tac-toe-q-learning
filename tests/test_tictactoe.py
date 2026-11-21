from pathlib import Path
import pytest
import numpy as np

from src.tictactoe import (
    AGENT, OPPONENT, QLearningAgent, available_actions, is_full,
    make_move, state_key, winner,
)


def test_empty_board():
    board = np.zeros((3, 3), dtype=int)
    assert available_actions(board) == list(range(9))
    assert winner(board) == 0
    assert not is_full(board)


def test_horizontal_win():
    board = np.array([[1, 1, 1], [-1, 0, -1], [0, 0, 0]])
    assert winner(board) == AGENT


def test_diagonal_loss():
    board = np.array([[-1, 0, 1], [0, -1, 1], [0, 0, -1]])
    assert winner(board) == OPPONENT


def test_make_move_does_not_mutate_original():
    board = np.zeros((3, 3), dtype=int)
    moved = make_move(board, 4, AGENT)
    assert board[1, 1] == 0
    assert moved[1, 1] == AGENT


def test_agent_explores_then_exploits():
    agent = QLearningAgent(seed=7)
    agent.epsilon = 1.0
    board = np.zeros((3, 3), dtype=int)
    action = agent.choose_action(board, training=True)
    assert action in available_actions(board)

    agent.epsilon = 0.0
    board = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    agent.q_values[state_key(board)][1] = 0.1
    agent.q_values[state_key(board)][2] = 0.9
    assert agent.choose_action(board, training=False) == 2
