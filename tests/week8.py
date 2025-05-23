#!/usr/bin/env python3

from typing import Iterable

Game = ...
InfoSet = ...
Player = ...
Strategy = ...
Regrets = ...

def update_counterfactual_regrets(
    game: Game, player: Player, strategies: dict[Player, Strategy],
    regrets: dict[InfoSet, Regrets], naive: bool
) -> dict[InfoSet, Regrets]:
    """ Update accumulated regrets with new regrets computed using given strategies.

    Parameters
    ----------
    game : Game
        An object representing the game tree
    player : Player
        The player whose regrets we want to update
    strategies : dict[Player, Strategy]
        A dictionary mapping each player to their behavioural strategy
    regrets : dict[InfoSet, Regrets]
        A dictionary mapping each information set to regrets over the available actions
    naive : bool
        Whether to weight the information set (action)-values by the counterfactual reach or not

    Returns
    -------
    dict[InfoSet, Regrets]
        An updated dictionary mapping information sets to regrets
    """

    raise NotImplementedError


def cfr(game: Game, epochs: int, naive: bool = False) -> Iterable[dict[Player, Strategy]]:
    """ Run the CFR algorithm to find Nash equilibrium in a given game.

    Parameters
    ----------
    game : Game
        An object representing the game tree
    epochs : int
        The number of epochs to run the algorithm for
    naive : bool
        Whether to weight the information set (action)-values by the counterfactual reach or not

    Returns
    -------
    Iterable[dict[Player, Strategy]]:
        A sequence of average strategy profiles produced by the algorithm
    """

    raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
