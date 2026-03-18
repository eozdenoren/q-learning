"""
Substitutability experiment: how does k2 affect collusion?

Trains Q-learning duopoly at gamma=0.95 for various k2 values.
k2 controls product substitutability:
  - Low k2 (e.g. 0.1): very differentiated, small collusion premium
  - High k2 (e.g. 0.9): nearly identical, large collusion premium

Writes results incrementally to substitutability_experiments.csv.
"""

import numpy as np
import csv
import os
import time

# ── Parameters ──────────────────────────────────────────────────────
K2_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67, 0.75, 0.85, 0.9]
GAMMA = 0.95
RUNS_PER_K2 = 10
ALPHA = 0.15
BETA = 4e-6
K1 = 9.0
C = 3.0
M = 15
CHECK_EVERY = 1000
STABLE_REQUIRED = 100_000
MAX_PERIODS = 5_000_000

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "substitutability_experiments.csv")

# ── Core functions ──────────────────────────────────────────────────

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

def log(msg):
    print(msg, flush=True)


# ── Training ────────────────────────────────────────────────────────

def train(gamma, seed, k2):
    prices, p_e, p_c, profit_e, profit_c = calculate_prices(K1, k2, C, M)
    n_actions = len(prices)
    n_states = n_actions * n_actions
    rng = np.random.default_rng(seed)
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    s = rng.integers(0, n_states)
    prev_pi1 = greedy_map(Q1)
    prev_pi2 = greedy_map(Q2)
    stable_count = 0
    step_count = 0
    converged = False

    while step_count < MAX_PERIODS:
        step_count += 1
        eps = epsilon_at(step_count, BETA)
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
        pi1 = profit1(p1_next, p2_next, C, K1, k2)
        pi2 = profit2(p1_next, p2_next, C, K1, k2)
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


# ── Simulation ──────────────────────────────────────────────────────

def simulate(Q1, Q2, start_p1, start_p2, prices, max_steps=200):
    rng_sim = np.random.default_rng(0)
    s = state_index(start_p1, start_p2, prices)
    first_seen = {}
    trajectory = []
    for t in range(max_steps):
        if s in first_seen:
            loop_start = first_seen[s]
            return trajectory, loop_start
        p1, p2 = index_to_state(s, prices)
        first_seen[s] = t
        trajectory.append((p1, p2))
        a1 = argmax_tie(Q1[s], rng_sim)
        a2 = argmax_tie(Q2[s], rng_sim)
        s = state_index(prices[a1], prices[a2], prices)
    return trajectory, -1


# ── Main ────────────────────────────────────────────────────────────

def main():
    columns = [
        "k2", "gamma", "seed", "converged", "steps",
        "p_e", "p_c", "profit_e", "profit_c",
        "collusion_premium_pct",
        "cycle_prices", "cycle_length",
        "avg_price_1", "avg_price_2", "avg_profit_1", "avg_profit_2",
        "delta_1", "delta_2",
    ]

    csv_file = open(OUT_FILE, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()
    csv_file.flush()

    total = len(K2_VALUES) * RUNS_PER_K2
    count = 0
    t_global = time.time()

    for k2 in K2_VALUES:
        prices, p_e, p_c, profit_e, profit_c = calculate_prices(K1, k2, C, M)
        collusion_premium = 100 * (profit_c - profit_e) / profit_e if profit_e > 0 else 0
        log(f"\n--- k2={k2}, p_e={p_e:.3f}, p_c={p_c:.3f}, "
            f"profit_e={profit_e:.2f}, profit_c={profit_c:.2f}, "
            f"premium={collusion_premium:.1f}% ---")

        for run in range(RUNS_PER_K2):
            count += 1
            seed = int.from_bytes(os.urandom(3), "big") % 1_000_000
            t0 = time.time()
            Q1, Q2, prices_t, converged, steps = train(GAMMA, seed, k2)
            elapsed = time.time() - t0

            # Find cycle from middle of grid
            mid = len(prices_t) // 2
            traj, loop_start = simulate(Q1, Q2, prices_t[mid], prices_t[mid], prices_t)

            if loop_start >= 0:
                cycle = traj[loop_start:]
                cycle_str = " -> ".join(f"({p1},{p2})" for p1, p2 in cycle)
                avg_p1 = np.mean([p1 for p1, p2 in cycle])
                avg_p2 = np.mean([p2 for p1, p2 in cycle])
                avg_pi1 = np.mean([profit1(p1, p2, C, K1, k2) for p1, p2 in cycle])
                avg_pi2 = np.mean([profit2(p1, p2, C, K1, k2) for p1, p2 in cycle])
                delta1 = (avg_pi1 - profit_e) / (profit_c - profit_e) if profit_c != profit_e else 0
                delta2 = (avg_pi2 - profit_e) / (profit_c - profit_e) if profit_c != profit_e else 0
            else:
                cycle_str = "no cycle"
                avg_p1 = avg_p2 = avg_pi1 = avg_pi2 = delta1 = delta2 = float('nan')
                cycle = []

            log(f"  [{count}/{total}] k2={k2}, run={run+1}, seed={seed}, "
                f"conv={'Y' if converged else 'N'}, steps={steps:,}, "
                f"delta=({delta1:.2f},{delta2:.2f}), {elapsed:.0f}s")

            row = {
                "k2": k2,
                "gamma": GAMMA,
                "seed": seed,
                "converged": converged,
                "steps": steps,
                "p_e": round(p_e, 3),
                "p_c": round(p_c, 3),
                "profit_e": round(profit_e, 2),
                "profit_c": round(profit_c, 2),
                "collusion_premium_pct": round(collusion_premium, 1),
                "cycle_prices": cycle_str,
                "cycle_length": len(cycle),
                "avg_price_1": round(avg_p1, 3) if not np.isnan(avg_p1) else "",
                "avg_price_2": round(avg_p2, 3) if not np.isnan(avg_p2) else "",
                "avg_profit_1": round(avg_pi1, 2) if not np.isnan(avg_pi1) else "",
                "avg_profit_2": round(avg_pi2, 2) if not np.isnan(avg_pi2) else "",
                "delta_1": round(delta1, 3) if not np.isnan(delta1) else "",
                "delta_2": round(delta2, 3) if not np.isnan(delta2) else "",
            }
            writer.writerow(row)
            csv_file.flush()

    csv_file.close()
    total_time = time.time() - t_global
    log(f"\nDone! {count} results in {total_time:.0f}s -> {OUT_FILE}")

    # Summary
    import pandas as pd
    df = pd.read_csv(OUT_FILE)
    log(f"\n{'='*80}")
    log("SUBSTITUTABILITY ANALYSIS (k2 vs Delta)")
    log(f"{'='*80}")
    log(f"\n{'k2':>6} | {'premium%':>8} | {'Avg D1':>7} | {'Avg D2':>7} | {'Std':>6} | {'Conv':>4}")
    log("-" * 55)
    for k2 in K2_VALUES:
        sub = df[df["k2"] == k2]
        if sub.empty:
            continue
        d1 = pd.to_numeric(sub["delta_1"], errors="coerce")
        d2 = pd.to_numeric(sub["delta_2"], errors="coerce")
        prem = sub["collusion_premium_pct"].iloc[0]
        conv = sub["converged"].sum()
        log(f"{k2:>6.2f} | {prem:>7.1f}% | {d1.mean():>7.2f} | {d2.mean():>7.2f} | "
            f"{d1.std():>6.2f} | {conv:>2}/{len(sub)}")


if __name__ == "__main__":
    main()
