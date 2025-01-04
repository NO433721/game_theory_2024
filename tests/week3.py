#!/usr/bin/env python3

from typing import Iterable
import numpy as np
from matplotlib import pyplot as plt



def compute_deltas(
    row_strategy: np.ndarray, col_strategy: np.ndarray,
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ Compute players' incentives to deviate from their strategies.

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
        An incentive to deviate for each player
    """
    row_expected_payoff=row_strategy@row_matrix@col_strategy
    row_best_response_payoff=max(row_matrix@col_strategy)
    row_delta=row_best_response_payoff-row_expected_payoff

    col_expected_payoff=row_strategy@col_matrix@col_strategy
    col_best_response_payoff=max(row_strategy@col_matrix)
    col_delta=col_best_response_payoff-col_expected_payoff

    return row_delta, col_delta
    #raise NotImplementedError


def compute_nash_conv(
    row_strategy: np.ndarray, col_strategy: np.ndarray,
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> float:
    """ Compute the NashConv value of a given strategy profile.

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
    float
        The NashConv value of the given strategy profile
    """
    return sum(compute_deltas(row_strategy, col_strategy, row_matrix, col_matrix))



def compute_exploitability(
    row_strategy: np.ndarray, col_strategy: np.ndarray,
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> float:
    """ Compute the exploitability of a given strategy profile.

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
    float
        The exploitability value of the given strategy profile
    """
    return compute_nash_conv(row_strategy, col_strategy, row_matrix, col_matrix)/2
    #raise NotImplementedError


def compute_epsilon(
    row_strategy: np.ndarray, col_strategy: np.ndarray,
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> float:
    """ Compute the epsilon of a given epsilon-Nash equilibrium strategy profile.

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
    float
        The epsilon value
    """
    return max(compute_deltas(row_strategy, col_strategy, row_matrix, col_matrix))
    #raise NotImplementedError


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, epochs: int, naive: bool
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """ Run the Fictitious Play algorithm to find a Nash equilibrium in a given game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    epochs : int
        The number of epochs to run the algorithm for
    naive : bool
        Whether to calculate the best response against the last
        opponent's strategy or the mean opponent's strategy

    Returns
    -------
    Iterable[tuple[np.ndarray, np.ndarray]]
        A sequence of average strategy profiles produced by the algorithm
    """

    avg_strategies=[]

    row_strategy=np.ones(row_matrix.shape[0])/row_matrix.shape[0]
    col_strategy=np.ones(col_matrix.shape[1])/col_matrix.shape[1]

    freq_row=np.zeros(row_matrix.shape[0])
    freq_col=np.zeros(col_matrix.shape[1])

    for epoch in range(epochs):
        freq_row += row_strategy
        freq_col += col_strategy

        avg_row_strategy= freq_row / (epoch + 1)
        avg_col_strategy= freq_col / (epoch + 1)
        avg_strategies.append((avg_row_strategy, avg_col_strategy))

        if naive:
            best_response_row=np.argmax(row_matrix @ col_strategy)
            best_response_col=np.argmax(row_strategy @ col_matrix)
        else:
            best_response_row=np.argmax(row_matrix @ avg_col_strategy)
            best_response_col=np.argmax(avg_row_strategy @ col_matrix)

        row_strategy=np.zeros(row_matrix.shape[0])
        row_strategy[best_response_row]=1

        col_strategy=np.zeros(col_matrix.shape[1])
        col_strategy[best_response_col]=1

    return avg_strategies

    #raise NotImplementedError


def plot_exploitability(
    row_matrix: np.ndarray, col_matrix: np.ndarray,
    strategies: Iterable[tuple[np.ndarray, np.ndarray]], label: str
) -> list[float]:
    """ Compute and plot the exploitability of a sequence of strategy profiles.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    strategies : Iterable[tuple[np.ndarray, np.ndarray]]
        The sequence of strategy profiles
    label : str
        The name of the algorithm that produced `strategies`

    Returns
    -------
    list[float]
        A sequence of exploitability values, one for each strategy profile
    """
    exploitabilities = []

    for row_strategy, col_strategy in strategies:
        exploitabilities.append(compute_exploitability(row_strategy, col_strategy, row_matrix, col_matrix))

    plt.plot(exploitabilities, label=label)

    return exploitabilities



    #raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
