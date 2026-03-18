"""
Punishment strategy analysis for Q-learning pricing.

For each trained model that converges to a collusive fixed point,
simulates what happens when firm 1 deviates to every lower price
while firm 2 stays at the cycle price.

Usage:
    python3 run_punishment.py
"""

import numpy as np
import csv
import os
import sys
import time

# ── Parameters (same as main experiment) ────────────────────────────
GAMMAS = [0.95, 0.8]  # Focus on collusive gammas
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

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "punishment_experiments.csv")

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

def train(gamma, seed):
    prices, p_e, p_c, profit_e, profit_c = calculate_prices(K1, K2, C, M)
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


# ── Simulation ──────────────────────────────────────────────────────

def simulate(Q1, Q2, start_p1, start_p2, prices, max_steps=200):
    """Follow greedy policy, return list of (p1, p2) tuples until cycle or max_steps."""
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


def format_trajectory(traj):
    return " → ".join(f"({p1},{p2})" for p1, p2 in traj)


# ── Main ────────────────────────────────────────────────────────────

def main():
    prices, p_e, p_c, profit_e, profit_c = calculate_prices(K1, K2, C, M)

    log(f"Prices: {prices}")
    log(f"p_e = {p_e:.3f}, p_c = {p_c:.3f}")
    log(f"profit_e = {profit_e:.2f}, profit_c = {profit_c:.2f}")
    log("")

    columns = [
        "gamma", "seed", "steps", "cycle_price",
        "deviation_p1", "deviation_profit_1shot",
        "cycle_profit", "1shot_gain",
        "returns_to_cycle", "punishment_length", "trajectory",
    ]

    csv_file = open(OUT_FILE, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()
    csv_file.flush()

    run_count = 0

    for gamma in GAMMAS:
        for run in range(RUNS_PER_GAMMA):
            seed = int.from_bytes(os.urandom(3), "big") % 1_000_000
            t0 = time.time()
            Q1, Q2, prices_t, converged, steps = train(gamma, seed)
            train_time = time.time() - t0

            if not converged:
                log(f"  gamma={gamma}, seed={seed}: did not converge, skipping")
                continue

            # Find the cycle — try multiple starting points
            found_fp = False
            cycle_price = None
            for sp in prices_t:
                traj, loop_start = simulate(Q1, Q2, sp, sp, prices_t)
                if loop_start < 0:
                    continue
                cycle = traj[loop_start:]
                if len(cycle) == 1 and cycle[0][0] == cycle[0][1]:
                    cycle_price = cycle[0][0]
                    found_fp = True
                    break

            if not found_fp:
                # Debug: show what we actually found
                traj, loop_start = simulate(Q1, Q2, prices_t[0], prices_t[0], prices_t)
                if loop_start >= 0:
                    cycle = traj[loop_start:]
                    log(f"  gamma={gamma}, seed={seed}: got cycle len={len(cycle)}: "
                        f"{format_trajectory(cycle[:5])}, skipping")
                else:
                    log(f"  gamma={gamma}, seed={seed}: no cycle detected at all, skipping")
                continue
            cycle_pi = profit1(cycle_price, cycle_price, C, K1, K2)
            cycle_idx = prices_t.index(cycle_price)

            log(f"gamma={gamma}, run={run+1}, seed={seed}, steps={steps:,}, "
                f"cycle=({cycle_price},{cycle_price}), {train_time:.0f}s")

            # Test deviations: firm 1 deviates to each lower price
            for dev_idx in range(cycle_idx):
                dev_price = prices_t[dev_idx]
                # One-shot deviation profit for firm 1
                dev_pi1 = profit1(dev_price, cycle_price, C, K1, K2)
                one_shot_gain = dev_pi1 - cycle_pi

                # Simulate from deviation state
                traj, loop_start_dev = simulate(Q1, Q2, dev_price, cycle_price, prices_t)

                # Check if it returns to the original cycle
                returns = False
                punishment_len = 0
                for i, (p1, p2) in enumerate(traj):
                    if p1 == cycle_price and p2 == cycle_price and i > 0:
                        returns = True
                        punishment_len = i  # steps until back to cycle
                        break

                run_count += 1
                row = {
                    "gamma": gamma,
                    "seed": seed,
                    "steps": steps,
                    "cycle_price": cycle_price,
                    "deviation_p1": dev_price,
                    "deviation_profit_1shot": round(dev_pi1, 2),
                    "cycle_profit": round(cycle_pi, 2),
                    "1shot_gain": round(one_shot_gain, 2),
                    "returns_to_cycle": returns,
                    "punishment_length": punishment_len,
                    "trajectory": format_trajectory(traj[:min(20, len(traj))]),
                }
                writer.writerow(row)
                csv_file.flush()

    csv_file.close()
    log(f"\nDone! {run_count} deviation experiments saved to {OUT_FILE}")

    # Print summary
    if run_count > 0:
        import pandas as pd
        df = pd.read_csv(OUT_FILE)

        log("\n" + "=" * 80)
        log("PUNISHMENT ANALYSIS")
        log("=" * 80)

        for gamma in GAMMAS:
            gdf = df[df["gamma"] == gamma]
            if gdf.empty:
                continue
            log(f"\ngamma = {gamma}:")
            log(f"  Models with symmetric fixed points: {gdf['seed'].nunique()}")
            log(f"  Total deviation tests: {len(gdf)}")
            log(f"  Returns to cycle: {gdf['returns_to_cycle'].sum()}/{len(gdf)} "
                f"({100*gdf['returns_to_cycle'].mean():.0f}%)")
            log(f"  Avg punishment length: {gdf.loc[gdf['returns_to_cycle'], 'punishment_length'].mean():.1f} steps")
            log(f"  Avg 1-shot gain from deviation: {gdf['1shot_gain'].mean():.2f}")

            # Show a few example trajectories
            log(f"\n  Example deviation trajectories:")
            for _, row in gdf.head(5).iterrows():
                log(f"    Deviate to {row['deviation_p1']}: "
                    f"gain={row['1shot_gain']:.1f}, "
                    f"returns={'Yes' if row['returns_to_cycle'] else 'No'}, "
                    f"punishment={row['punishment_length']} steps")
                log(f"      {row['trajectory']}")


if __name__ == "__main__":
    main()
