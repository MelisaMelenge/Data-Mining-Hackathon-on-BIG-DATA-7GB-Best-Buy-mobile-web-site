"""simulation_ca.py
Event-based Cellular Automata simulation prototype for Workshop #4.

Usage:
    python simulation_ca.py --steps 100 --rows 50 --cols 50 --output ca_counts.csv

This script runs a CA where each cell is a session state:
0 = inactive/abandoned, 1 = browsing, 2 = clicked, 3 = purchased
Transition probabilities depend on a per-cell 'query_quality' and neighbor influence.
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter

def initialize(rows, cols, seed=42):
    rng = np.random.RandomState(seed)
    state = np.zeros((rows,cols), dtype=int)
    state[rng.rand(rows, cols) < 0.7] = 1
    state[rng.rand(rows, cols) < 0.03] = 2
    state[rng.rand(rows, cols) < 0.01] = 3
    q_quality = rng.beta(1.2, 3.0, size=(rows,cols))
    return state, q_quality, rng

def step_ca(state, q_quality, rng):
    new = state.copy()
    rows, cols = state.shape
    for i in range(rows):
        for j in range(cols):
            s = state[i, j]
            qq = q_quality[i, j]
            neigh = state[max(0,i-1):min(rows,i+2), max(0,j-1):min(cols,j+2)]
            clicked_neighbors = np.sum((neigh == 2) | (neigh == 3)) - ((s==2) or (s==3))
            neigh_cells = neigh.size - 1
            ni = clicked_neighbors / max(1, neigh_cells)
            if s == 1:
                p_click = 0.05 + 0.4 * qq + 0.25 * ni
                p_abandon = 0.02 + 0.2 * (1-qq)
                r = rng.rand()
                if r < p_click:
                    new[i,j] = 2
                elif r < p_click + p_abandon:
                    new[i,j] = 0
            elif s == 2:
                p_buy = 0.02 + 0.3 * qq + 0.15 * ni
                if rng.rand() < p_buy:
                    new[i,j] = 3
    return new

def run(args):
    state, q_quality, rng = initialize(args.rows, args.cols, seed=args.seed)
    # optional perturbation: boost a few cells
    if args.perturb > 0:
        idxs = rng.choice(args.rows*args.cols, size=args.perturb, replace=False)
        for p in idxs:
            i, j = divmod(p, args.cols)
            q_quality[i,j] = 0.99
    counts_over_time = []
    cur = state.copy()
    for t in range(args.steps):
        counts_over_time.append(Counter(cur.flatten()))
        cur = step_ca(cur, q_quality, rng)
    df = pd.DataFrame([{
        'step': t,
        'inactive': c.get(0,0),
        'browsing': c.get(1,0),
        'clicked': c.get(2,0),
        'purchased': c.get(3,0)
    } for t, c in enumerate(counts_over_time)])
    df.to_csv(args.output, index=False)
    final = df.iloc[-1]
    purchased_frac = final['purchased'] / (args.rows*args.cols)
    print(f"Saved CA counts to {args.output}")
    print(f"Final purchased fraction: {purchased_frac:.4f}")
    print(df.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--rows', type=int, default=50)
    parser.add_argument('--cols', type=int, default=50)
    parser.add_argument('--perturb', type=int, default=8, help='number of cells to boost quality')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='ca_counts.csv')
    args = parser.parse_args()
    run(args)
