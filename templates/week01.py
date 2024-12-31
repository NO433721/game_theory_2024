#!/usr/bin/env python3
from typing import Tuple, Any

import numpy as np



def evaluate_pair(
    row_strategy: np.ndarray, col_strategy: np.ndarray,
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[Any, Any]:
    """ Compute the expected utility of each player in a general-sum game.

    Parameters
    ----------
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    # Compute the expected utility for the row player
    row_expected_utility = row_strategy @ row_matrix @ col_strategy

    # Compute the expected utility for the column player
    col_expected_utility = row_strategy @ col_matrix @ col_strategy

    return row_expected_utility, col_expected_utility

#raise NotImplementedError


def evaluate(
    row_strategy: np.ndarray, col_strategy: np.ndarray, matrix: np.ndarray
) -> tuple[Any, Any]:
    """ Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy
    matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    row_expected_utility = row_strategy @ matrix @ col_strategy
    col_expected_utility = -row_expected_utility

    return row_expected_utility, col_expected_utility

    #raise NotImplementedError



def evaluate_row_against_best_response(
    row_strategy: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ Compute the utilities when the row player plays against a best response strategy.

    Note that this function works only for zero-sum games

    Parameters
    ----------
    row_strategy : np.ndarray
        The row player's strategy
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    col_util=max(row_strategy@col_matrix)
    row_util=-col_util
    return row_util, col_util
    # raise NotImplementedError


def evaluate_col_against_best_response(
    col_strategy: np.ndarray, row_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ Compute the utilities when the column player plays against a best response strategy.

    Note that this function works only for zero-sum games

    Parameters
    ----------
    col_strategy : np.ndarray
        The column player's strategy
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    row_util = max(row_matrix@col_strategy)
    col_util = -row_util
    return row_util, col_util
    #raise NotImplementedError


def iterated_removal_of_dominated_actions(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Run the Iterated Removal of Dominated Actions algorithm.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A pair of reduced payoff matrices and a pair of remaining actions for each player
    """

    reduced_matrix1 = row_matrix
    reduced_matrix2 = col_matrix

    actions1 = [i for i in range(row_matrix.shape[0])]
    actions2 = [i for i in range(col_matrix.shape[1])]

    while True:
        row_remove = []
        col_remove = []

        for i in range(reduced_matrix1.shape[0]):
            result = np.all(reduced_matrix1[i, :].reshape(1, reduced_matrix1.shape[1]) <= reduced_matrix1, axis=1)
            result[i] = False

            if result.any():
                row_remove.append(i)
                actions1.pop(i)
                # print(actions1)

        # row_removes.append(row_remove)
        reduced_matrix1 = np.delete(reduced_matrix1, row_remove, axis=0)
        reduced_matrix2 = np.delete(reduced_matrix2, row_remove, axis=0)

        for i in range(reduced_matrix2.shape[1]):
            result = np.all(reduced_matrix2[:, i].reshape(reduced_matrix2.shape[0], 1) <= reduced_matrix2, axis=0)
            result[i] = False

            if result.any():
                col_remove.append(i)
                actions2.pop(i)
                # print(actions2)

        # col_removes.append(col_remove)
        reduced_matrix1 = np.delete(reduced_matrix1, col_remove, axis=1)
        reduced_matrix2 = np.delete(reduced_matrix2, col_remove, axis=1)

        if row_remove == [] and col_remove == []:
            break

    return reduced_matrix1, reduced_matrix2, actions1, actions2


def main() -> None:
    pass


if __name__ == '__main__':
    main()
