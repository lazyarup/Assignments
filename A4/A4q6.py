from pulp import *

# ---------------- data -----------------
sup = dict(F1=200, F2=160, F3=90)
dem = dict(W1=180, W2=120, W3=150)
cst = dict(
    F1=dict(W1=16, W2=20, W3=12),
    F2=dict(W1=14, W2=8,  W3=18),
    F3=dict(W1=26, W2=24, W3=16)
)

# ------------- model -------------------
prob = LpProblem('TP', LpMinimize)
x = LpVariable.dicts('x', (sup, dem), 0)

prob += lpSum(x[f][w] * cst[f][w] for f in sup for w in dem)
for f in sup: 
    prob += lpSum(x[f][w] for w in dem) == sup[f]
for w in dem: 
    prob += lpSum(x[f][w] for f in sup) == dem[w]

prob.solve()

# ------------- output ------------------
print('Status:', LpStatus[prob.status])
for f in sup:
    for w in dem:
        v = x[f][w].value()
        if v:  # print only non-zeros
            print(f'{f} → {w}: {v}')
print('Total cost:', value(prob.objective), 'BDT')
