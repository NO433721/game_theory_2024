#!/usr/bin/env python3

import numpy as np
from scipy.optimize import linprog



def find_nash_equilibrium(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Find a Nash equilibrium of a zero-sum normal-form game using linear programming.

    Parameters
    ----------
    matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A strategy profile that forms a Nash equilibrium
    """
    row_strategy=solve_for_nash_row(matrix)
    col_strategy=solve_for_nash_col(matrix)

    return row_strategy, col_strategy
    #raise NotImplementedError


def find_correlated_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray) -> np.ndarray:
    """
    Find a Correlated equilibrium of a normal-form game using linear programming.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A distribution over joint actions that forms a Correlated equilibrium
    """
    num_row=row_matrix.shape[0]
    num_col=col_matrix.shape[1]

    num_var=num_row*num_col
    A_ub=np.zeros((num_row*(num_row-1)+num_col*(num_col-1), num_var))
    A_ub=[]

    for a_r in range(num_row):
        for a_r_p in range(num_row):
            if a_r_p == a_r:
                continue

            constraint=[]
            constraint.extend([0]*a_r*num_col)

            for a_c in range(num_col):
                constraint.append(row_matrix[a_r,a_c]-row_matrix[a_r_p,a_c])
            constraint.extend([0]*(num_var-(a_r+1)*num_col))

            A_ub.append(constraint)

    for a_c in range(num_col):
        for a_c_p in range(num_col):
            if a_c_p == a_c:
                continue

            constraint=[0]*num_var
            for a_r in range(num_row):

                constraint[a_c + a_r * num_col]= col_matrix.T[a_c, a_r] - col_matrix.T[a_c_p,a_r]
            A_ub.append(constraint)

    A_ub=np.asarray(A_ub)
    #print(A_ub)

    c=np.zeros(num_var)

    A_ub=-A_ub
    b_ub=np.zeros((num_row*(num_row-1)+num_col*(num_col-1)))

    A_eq=np.ones((1,num_var))
    b_eq=1

    result = linprog(c, A_ub=A_ub, A_eq=A_eq, b_ub=b_ub, b_eq=b_eq, method="highs")

    if result.success:
        prob_matrix=np.asarray(result.x).reshape(num_row, num_col)
        return prob_matrix


def solve_for_nash_col(matrix: np.ndarray) -> np.ndarray:
    # for column player
    A_ub = np.asarray(matrix)
    A_ub = np.hstack((A_ub, -np.ones((A_ub.shape[0], 1))))
    #print(A_ub)

    A_eq = np.ones((1,A_ub.shape[1]))
    A_eq[-1,-1] = 0
    #print(A_eq)
    #print(A_eq.shape)
    # A_eq=np.vstack((A_eq, np.ones((1,A_eq.shape[1]))))
    # A_eq[-1,-1]=0
    # print(A_eq)
    #
    c = np.asarray([0] * (A_ub.shape[1] - 1) + [1])
    #print(c)
    #print(c.shape)
    # print(c)
    # b_eq=np.zeros(A_eq.shape[0])
    # b_eq[-1]=1
    # print(b_eq)
    b_ub = np.zeros(A_ub.shape[0])
    b_eq = np.ones((1,1))

    bounds = [(0, None)] * (A_ub.shape[1] - 1) + [(None, None)]
    #print(bounds)

    result = linprog(c, A_ub=A_ub, A_eq=A_eq, b_ub=b_ub, b_eq=b_eq, bounds=bounds, method="highs")
    #print(result)

    if result.success:
        return result.x[:-1]

def solve_for_nash_row(matrix: np.ndarray) -> np.ndarray:

    A_ub = np.asarray(matrix)
    A_ub=-A_ub.T
    #print(A_ub)
    #print(A_ub.shape)
    A_ub = np.hstack((A_ub, np.ones((A_ub.shape[0], 1))))
    #print(A_ub)

    A_eq = np.ones((1,A_ub.shape[1]))
    A_eq[-1,-1] = 0
    #print(A_eq)
    #print(A_eq.shape)
    # A_eq=np.vstack((A_eq, np.ones((1,A_eq.shape[1]))))
    # A_eq[-1,-1]=0
    # print(A_eq)
    #
    c = np.asarray([0] * (A_ub.shape[1] - 1) + [-1])
    #print(c)
    #print(c.shape)
    # print(c)
    # b_eq=np.zeros(A_eq.shape[0])
    # b_eq[-1]=1
    # print(b_eq)
    b_ub = np.zeros(A_ub.shape[0])
    b_eq = np.ones((1,1))
    #print(b_ub)
    #print(b_eq)

    bounds = [(0, None)] * (A_ub.shape[1] - 1) + [(None, None)]
    #print(bounds)

    result = linprog(c, A_ub=A_ub, A_eq=A_eq, b_ub=b_ub, b_eq=b_eq, bounds=bounds, method="highs")
    #print(result)

    if result.success:
        return result.x[:-1]


def main() -> None:
    pass


if __name__ == '__main__':
    main()

