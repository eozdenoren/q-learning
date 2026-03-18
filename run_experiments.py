"""
Batch experiment runner for Q-learning pricing simulations.
Bypasses Streamlit — runs training + simulation directly and outputs CSV.

Usage:
    python3 run_experiments.py
"""

import numpy as np
import pandas as pd
import os
import time

# ── Parameters ──────────────────────────────────────────────────────
GAMMAS = [0.95, 0.8, 0.7, 0.6, 0.5, 0.4]
RUNS_PER_GAMMA = 10
ALPHA = 0.15
BETA = 4e-6
K1 = 9.0
K2 = 0.67
C = 3.0
M = 15
CHECK_EVERY = 1000
STABLE_REQUIRED = 100_000
MAX_PERIODS = 5_000_000

# ── Core functions (copied from state_econ.py, no Streamlit dependency) ─

def demand1(p1, p2, k1, k2):
    return max(0.0, k1 - p1 + k2 * p2)

def demand2(p1, p2, k1, k2):
    return max(0.0, k1 - p2 + k2 * p1)

def profit1(p1, p2, c, k1, k2):
    return (p1 - c) * demand1(p1, p2, k1, k2)

def profit2(p1, p2, c, k1, k2):
    return (p2 - c) * demand2(p1, p2, k1, k2)

def calculate_prices(k1, k2, c, m):
    p_e = (k1 + c) / (2 - k2)
    profit_e = (p_e - c) * demand1(p_e, p_e, k1, k2)
    p_c = (2 * k1 + 2 * c * (1 - k2)) / (4 * (1 - k2))
    profit_c = (p_c - c) * demand1(p_c, p_c, k1, k2)
    price_start = 2 * p_e - p_c
    price_end = 2 * p_c - p_e
    prices = np.round(np.linspace(price_start, price_end, m), 3).tolist()
    return prices, p_e, p_c, profit_e, profit_c

def state_index(p1, p2, prices):
    price_to_idx = {p: i for i, p in enumerate(prices)}
    return price_to_idx[p1] * len(prices) + price_to_idx[p2]

def index_to_state(s, prices):
    n = len(prices)
    return prices[s // n], prices[s % n]

def argmax_tie(x, rng):
    m = np.max(x)
    idxs = np.flatnonzero(np.isclose(x, m))
    return int(rng.choice(idxs))

def greedy_map(Q):
    return np.argmax(Q, axis=1)

def epsilon_at(step, beta):
    return float(np.exp(-beta * step))


# ── Training ────────────────────────────────────────────────────────

def train(gamma, seed):
    """Train Q-learning agents. Returns (Q1, Q2, prices, converged, steps)."""
    prices, p_e, p_c, profit_e, profit_c = calculate_prices(K1, K2, C, M)
    n_actions = len(prices)
    n_states = n_actions * n_actions

    rng = np.random.default_rng(seed)
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))

    # Random starting state
    s = rng.integers(0, n_states)

    prev_pi1 = greedy_map(Q1)
    prev_pi2 = greedy_map(Q2)
    stable_count = 0
    step_count = 0
    converged = False

    while step_count < MAX_PERIODS:
        step_count += 1
        eps = epsilon_at(step_count, BETA)

        # ε-greedy action selection
        if rng.random() < eps:
            a1 = rng.integers(0, n_actions)
        else:
            a1 = argmax_tie(Q1[s], rng)

        if rng.random() < eps:
            a2 = rng.integers(0, n_actions)
        else:
            a2 = argmax_tie(Q2[s], rng)

        p1_next = prices[a1]
        p2_next = prices[a2]
        s_next = state_index(p1_next, p2_next, prices)

        pi1 = profit1(p1_next, p2_next, C, K1, K2)
        pi2 = profit2(p1_next, p2_next, C, K1, K2)

        Q1[s, a1] = (1 - ALPHA) * Q1[s, a1] + ALPHA * (pi1 + gamma * np.max(Q1[s_next]))
        Q2[s, a2] = (1 - ALPHA) * Q2[s, a2] + ALPHA * (pi2 + gamma * np.max(Q2[s_next]))

        s = s_next

        if step_count % CHECK_EVERY == 0:
            current_pi1 = greedy_map(Q1)
            current_pi2 = greedy_map(Q2)
            if np.array_equal(current_pi1, prev_pi1) and np.array_equal(current_pi2, prev_pi2):
                stable_count += CHECK_EVERY
                if stable_count >= STABLE_REQUIRED:
                    converged = True
                    break
            else:
                stable_count = 0
                prev_pi1 = current_pi1
                prev_pi2 = current_pi2

    return Q1, Q2, prices, converged, step_count


# ── Simulation (greedy, no exploration) ─────────────────────────────

def simulate(Q1, Q2, start_p1, start_p2, prices, max_steps=10000):
    """Follow greedy policy from (start_p1, start_p2) until cycle detected.
    Returns dict with cycle info."""
    rng_sim = np.random.default_rng(0)  # deterministic for simulation
    s = state_index(start_p1, start_p2, prices)

    first_seen = {}
    path_states = []  # list of state indices
    path_prices = []  # list of (p1, p2)

    for t in range(max_steps):
        if s in first_seen:
            loop_start = first_seen[s]
            cycle_states = path_prices[loop_start:]
            transient = path_prices[:loop_start]
            return {
                "cycle": cycle_states,
                "transient": transient,
                "cycle_len": len(cycle_states),
            }
        first_seen[s] = t
        p1, p2 = index_to_state(s, prices)
        path_states.append(s)
        path_prices.append((p1, p2))

        a1 = argmax_tie(Q1[s], rng_sim)
        a2 = argmax_tie(Q2[s], rng_sim)
        p1_next = prices[a1]
        p2_next = prices[a2]
        s = state_index(p1_next, p2_next, prices)

    # No cycle found
    return {"cycle": path_prices[-10:], "transient": path_prices[:-10], "cycle_len": -1}


def format_prices(price_list):
    """Format list of (p1,p2) tuples as string."""
    return " → ".join(f"({p1},{p2})" for p1, p2 in price_list)


# ── Main ────────────────────────────────────────────────────────────

def main():
    prices, p_e, p_c, profit_e, profit_c = calculate_prices(K1, K2, C, M)
    n_prices = len(prices)

    # Find nearest grid prices to p_e and p_c
    p_e_grid = min(prices, key=lambda x: abs(x - p_e))
    p_c_grid = min(prices, key=lambda x: abs(x - p_c))

    # Three starting conditions
    start_configs = [
        ("Nash-Nash", p_e_grid, p_e_grid),
        ("Collusion-Collusion", p_c_grid, p_c_grid),
        ("Collusion-Nash", p_c_grid, p_e_grid),
    ]

    print(f"Price grid: {prices}")
    print(f"p_e = {p_e:.3f} (nearest grid: {p_e_grid})")
    print(f"p_c = {p_c:.3f} (nearest grid: {p_c_grid})")
    print(f"profit_e = {profit_e:.2f}, profit_c = {profit_c:.2f}")
    print(f"Gammas: {GAMMAS}")
    print(f"Runs per gamma: {RUNS_PER_GAMMA}")
    print(f"Starting configs: {[s[0] for s in start_configs]}")
    print(f"Total experiments: {len(GAMMAS) * RUNS_PER_GAMMA * len(start_configs)}")
    print()

    results = []
    run_id = 0

    for gamma in GAMMAS:
        for run in range(RUNS_PER_GAMMA):
            seed = int.from_bytes(os.urandom(3), "big") % 1_000_000
            t0 = time.time()
            Q1, Q2, prices, converged, steps = train(gamma, seed)
            train_time = time.time() - t0

            status = "converged" if converged else "MAX_PERIODS"
            print(f"gamma={gamma}, run={run+1}/{RUNS_PER_GAMMA}, "
                  f"seed={seed}, steps={steps:,}, {status}, "
                  f"{train_time:.1f}s")

            for start_name, sp1, sp2 in start_configs:
                run_id += 1
                sim = simulate(Q1, Q2, sp1, sp2, prices)

                cycle = sim["cycle"]
                cycle_len = sim["cycle_len"]

                # Compute average prices and profits over cycle
                if cycle:
                    avg_p1 = np.mean([p[0] for p in cycle])
                    avg_p2 = np.mean([p[1] for p in cycle])
                    avg_pi1 = np.mean([profit1(p[0], p[1], C, K1, K2) for p in cycle])
                    avg_pi2 = np.mean([profit2(p[0], p[1], C, K1, K2) for p in cycle])
                else:
                    avg_p1 = avg_p2 = avg_pi1 = avg_pi2 = float("nan")

                delta1 = (avg_pi1 - profit_e) / (profit_c - profit_e) if profit_c != profit_e else 0
                delta2 = (avg_pi2 - profit_e) / (profit_c - profit_e) if profit_c != profit_e else 0

                cycle_str = format_prices(cycle)
                # Transient: include starting point + key transitions
                transient = sim["transient"]
                if transient:
                    # Compress: show starting point, then unique transitions
                    seen = set()
                    compressed = []
                    for p in transient:
                        if p not in seen:
                            seen.add(p)
                            compressed.append(p)
                    transient_str = format_prices(compressed)
                else:
                    transient_str = ""

                path_str = transient_str + (" → " if transient_str else "") + cycle_str

                results.append({
                    "run": run_id,
                    "gamma": gamma,
                    "seed": seed,
                    "start_config": start_name,
                    "start_p1": sp1,
                    "start_p2": sp2,
                    "converged": converged,
                    "steps": steps,
                    "train_time_s": round(train_time, 1),
                    "cycle_len": cycle_len,
                    "cycle": cycle_str,
                    "avg_p1": round(avg_p1, 2),
                    "avg_p2": round(avg_p2, 2),
                    "avg_pi1": round(avg_pi1, 2),
                    "avg_pi2": round(avg_pi2, 2),
                    "Delta1": round(delta1, 3),
                    "Delta2": round(delta2, 3),
                    "path": path_str,
                })

    # Save results
    df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_experiments.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Average Delta by gamma and starting config")
    print("=" * 80)
    summary = df.groupby(["gamma", "start_config"])[["Delta1", "Delta2"]].agg(["mean", "std"])
    print(summary.to_string())

    print("\n" + "=" * 80)
    print("SUMMARY: Cycle lengths by gamma")
    print("=" * 80)
    cycle_summary = df.groupby("gamma")["cycle_len"].value_counts().unstack(fill_value=0)
    print(cycle_summary.to_string())

    print("\n" + "=" * 80)
    print("SUMMARY: Convergence by gamma")
    print("=" * 80)
    conv_summary = df.groupby("gamma").agg(
        converged_pct=("converged", "mean"),
        avg_steps=("steps", "mean"),
        avg_time=("train_time_s", "mean"),
    )
    print(conv_summary.to_string())


if __name__ == "__main__":
    main()
