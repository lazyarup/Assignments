import numpy as np, pandas as pd
from scipy.optimize import linprog

#-------------------------------------------------------------
#  Simplex (max c^T x,  A x ≤ b,  x ≥ 0)
#-------------------------------------------------------------
def simplex(c, A, b, show=True):
    m, n  = A.shape
    T     = np.zeros((m + 1, n + m + 1))               # tableau
    T[:m, :n], T[:m, n:n+m], T[:m, -1] = A, np.eye(m), b
    T[-1, :n] = -c                                     # objective row

    names  = [*(f'x{i+1}' for i in range(n)),
              *(f's{i+1}' for i in range(m)), 'RHS']
    basic  = list(range(n, n + m))                     # start with slacks
    step   = 0

    while True:
        if show:                                       # pretty print step
            print(f'\n{"="*50}\nSTEP {step}\n{"="*50}')
            idx = [names[i] for i in basic] + ['Z']
            print(pd.DataFrame(T, index=idx, columns=names).round(2))

        col = np.argmin(T[-1, :-1])                    # entering variable
        if T[-1, col] >= 0:
            break                      # optimal found

        ratios = np.where(T[:-1, col] > 0,
                           T[:-1, -1] / T[:-1, col],
                           np.inf)
        row = ratios.argmin()                          # leaving variable
        if np.isinf(ratios[row]): 
            return None, None    # unbounded

        T[row] /= T[row, col]                          # pivot
        for r in range(m + 1):
            if r != row:
                T[r] -= T[r, col] * T[row]
        basic[row] = col
        step += 1

    sol = np.zeros(n)
    for r, c_idx in enumerate(basic):
        if c_idx < n:
            sol[c_idx] = T[r, -1]

    return sol, T[-1, -1]

c = np.array([12, 15, 14])
A = np.array([[1, 1, 1],
                [0.02, 0.04, 0.03],
                [3, 2, 5]])
b = np.array([100, 3, 300])

print("COAL BLENDING PROBLEM – SIMPLEX METHOD")
x, profit = simplex(c, A, b)

print(f"\nOptimal mix (tons): {x.astype(int)}")
print(f"Maximum profit:      {int(profit)} BDT")

# SciPy verification
res = linprog(c=-c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
print("\nSciPy check:", res.x.astype(int), "profit =", int(-res.fun), "BDT")
