#!/usr/bin/env python3
from itertools import combinations
from typing import Iterable
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog


def best_response_value_func(matrix: np.ndarray, step_size: float) -> None:
    """ Plot the best response value function of the row player in a 2xN zero-sum matrix game.

    Parameters
    ----------
    matrix : np.ndarray
        The row player's payoff matrix
    step_size : float
        The step size for the probability of the first action of the row player
    """
    probability = np.linspace(0, 1, int(1 / step_size) + 1)
    best_response_values = []
    for prob in probability:
        mixed_strategy = np.array([prob, 1 - prob])

        expected_payoffs = mixed_strategy @ matrix

        best_response = np.min(expected_payoffs)
        best_response_values.append(best_response)

    plt.figure(figsize=(8, 5))
    plt.scatter(probability, best_response_values, label="Best Response Value", color="cyan", edgecolor="k", s=50)
    plt.xlabel("Probability of First Action of Row player", fontsize=12)
    plt.ylabel("Utility of the Row Player", fontsize=12)
    plt.title("Best Response Values for Row Player", fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()


#raise NotImplementedError


def verify_support_one_side(
    row_support: np.ndarray|list[int], col_support: np.ndarray|list[int], matrix: np.ndarray
) -> np.ndarray | None:
    """ Construct a set of linear equations to check whether there exists a Nash equilibrium

    Parameters
    ----------
    row_support : np.ndarray
        The row player's support
    col_support : np.ndarray
        The column player's support
    matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    np.ndarray | None
        The column player's strategy, if it exists, otherwise `None`
    """
    row_support=np.asarray(row_support)
    col_support=np.asarray(col_support)

    sub_matrix = matrix[np.ix_(row_support, col_support)]

    sub_matrix=np.hstack((sub_matrix, np.ones(sub_matrix.shape[0]).reshape(-1,1)))
    sub_matrix=np.vstack((sub_matrix, np.ones(sub_matrix.shape[1])))

    sub_matrix[-1, -1]=0


    A_eq=sub_matrix
    b_eq=np.zeros(sub_matrix.shape[0])
    b_eq[-1]=1

    c = np.zeros(A_eq.shape[1])

    bounds = [(0, None) for _ in range(A_eq.shape[1])]
    bounds[-1]=(None, None)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        col_strategy = np.zeros(matrix.shape[1])
        col_strategy[col_support] = result.x[:-1] # Exclude the augmented variable
        constant=result.x[-1]

        for i in range(matrix.shape[0]):
            if i not in row_support and matrix[i,:]@col_strategy>constant:
                return None

        return col_strategy
    else:
        return None


    # raise NotImplementedError


def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """ Run the Support Enumeration algorithm to find all Nash equilibria in a given game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    Iterable[tuple[np.ndarray, np.ndarray]]
        A sequence of strategy profiles corresponding to found Nash equilibria
    """
    num_row_strategies, num_col_strategies = row_matrix.shape
    nash_equilibria = []

    # Iterate over all possible support sizes for both players
    for row_support_size in range(1, num_row_strategies + 1):
        for col_support_size in range(1, num_col_strategies + 1):
            # Iterate over all possible supports for both players
            for row_support in combinations(range(num_row_strategies), row_support_size):
                for col_support in combinations(range(num_col_strategies), col_support_size):

                    if len(row_support) == 1 or len(col_support) == 1:
                        if len(col_support) != 1 or len(row_support) != 1:
                            continue

                        if np.argmax(col_matrix[row_support, :]) != col_support:
                            continue

                        if np.argmax(row_matrix[:, col_support]) != row_support:
                            continue

                        else:
                            row_strategy = np.zeros(row_matrix.shape[0])
                            row_strategy[row_support] = 1

                            col_strategy = np.zeros(col_matrix.shape[1])
                            col_strategy[col_support] = 1

                            nash_equilibria.append((row_strategy, col_strategy))

                    else:
                        row_strategy=verify_support_one_side(col_support, row_support, col_matrix.T)
                        col_strategy=verify_support_one_side(row_support, col_support, row_matrix)

                        if row_strategy is not None and col_strategy is not None:
                            nash_equilibria.append((row_strategy, col_strategy))

    return nash_equilibria
    #raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
