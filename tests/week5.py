#!/usr/bin/env python3

from typing import Iterable
import numpy as np


def regret_minimization(
    row_matrix: np.ndarray, col_matrix: np.ndarray, epochs: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """ Run the Regret Minimization algorithm to find a Nash equilibrium in a given game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    epochs : int
        The number of epochs to run the algorithm for

    Returns
    -------
    Iterable[tuple[np.ndarray, np.ndarray]]
        A sequence of average strategy profiles produced by the algorithm
    """
    #regret matching

    # Initialize strategies with uniform probabilities
    num_row_actions = row_matrix.shape[0]
    num_col_actions = col_matrix.shape[1]

    row_strategy = np.ones(num_row_actions) / num_row_actions
    col_strategy = np.ones(num_col_actions) / num_col_actions

    # Cumulative regret for each action
    cumulative_regret_row = np.zeros(num_row_actions)
    cumulative_regret_col = np.zeros(num_col_actions)

    # Average strategies over time
    avg_row_strategy = np.zeros(num_row_actions)
    avg_col_strategy = np.zeros(num_col_actions)
    avg_strategies=[]

    for epoch in range(1, epochs + 1):
        # Calculate expected payoffs for current strategies
        row_payoffs = row_matrix @ col_strategy  # Payoffs for each row action
        col_payoffs = row_strategy @ col_matrix  # Payoffs for each column action

        expected_payoff_row = np.dot(row_strategy, row_payoffs)
        expected_payoff_col = np.dot(col_strategy, col_payoffs) #!

        # Calculate best response payoffs
        best_response_row = np.max(row_payoffs)
        best_response_col = np.max(col_payoffs)

        # Update cumulative regrets
        cumulative_regret_row += row_payoffs - expected_payoff_row
        cumulative_regret_col += col_payoffs - expected_payoff_col

        # Ensure positive regrets only
        positive_regret_row = np.maximum(cumulative_regret_row, 0)
        positive_regret_col = np.maximum(cumulative_regret_col, 0)

        # Update strategies proportional to positive regrets
        if np.sum(positive_regret_row) > 0:
            row_strategy = positive_regret_row / np.sum(positive_regret_row)
        else:
            row_strategy = np.ones(num_row_actions) / num_row_actions  # Uniform if no regrets / no incentive to deviate

        if np.sum(positive_regret_col) > 0:
            col_strategy = positive_regret_col / np.sum(positive_regret_col)
        else:
            col_strategy = np.ones(num_col_actions) / num_col_actions  # Uniform if no regrets

        # Update average strategies
        avg_row_strategy = (avg_row_strategy * (epoch - 1) + row_strategy) / epoch
        avg_col_strategy = (avg_col_strategy * (epoch - 1) + col_strategy) / epoch

        avg_strategies.append((avg_row_strategy, avg_col_strategy))
    #raise NotImplementedError
    return avg_strategies

def main() -> None:
    pass


if __name__ == '__main__':
    main()
