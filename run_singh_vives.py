"""
Singh-Vives demand Q-learning experiments.

Microfounded demand from representative consumer utility:
  U = alpha*(q1+q2) - 0.5*(q1^2 + 2*gamma*q1*q2 + q2^2)

Yields demand: q_i = (a - p_i + gamma*p_j) / (1 - gamma^2)
where a = alpha*(1 - gamma).

Key properties:
  gamma = 0 → independent monopolies (p_e = p_c = (alpha+c)/2)
  gamma → 1 → homogeneous Bertrand (p_e → c, Bertrand paradox)
  p_c = (alpha+c)/2 is CONSTANT for all gamma

Parameters: alpha=12, c=3, m=15, discount_factor=0.95
Sweep gamma (substitutability) from 0.1 to 0.95.
"""

import numpy as np
import csv
import os
import time

# ── Parameters ──────────────────────────────────────────────────────
ALPHA_DEMAND = 12.0  # demand intercept parameter (not learning rate!)
C = 3.0
M = 15
GAMMAS_DEMAND = [0.1, 0.3, 0.5, 0.6, 0.67, 0.8, 0.9, 0.95]
DISCOUNT_FACTOR = 0.95
RUNS_PER_GAMMA = 10

# Q-learning parameters
ALPHA_LR = 0.15       # learning rate
BETA_EXPLORE = 4e-6   # exploration decay
CHECK_EVERY = 1000
STABLE_REQUIRED = 100_000
MAX_PERIODS = 5_000_000

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_sv.csv")


# ── Demand and profit functions ─────────────────────────────────────

def sv_demand(p_i, p_j, alpha, gamma):
    """Singh-Vives demand for firm i."""
    a = alpha * (1 - gamma)
    return max(0.0, (a - p_i + gamma * p_j) / (1 - gamma**2))

def sv_profit(p_i, p_j, c, alpha, gamma):
    return (p_i - c) * sv_demand(p_i, p_j, alpha, gamma)

def sv_prices_and_benchmarks(alpha, gamma, c, m):
    """Compute Nash price, collusive price, profits, and price grid."""
    a = alpha * (1 - gamma)
    p_e = (a + c) / (2 - gamma)
    p_c = (alpha + c) / 2  # constant for all gamma!

    profit_e = sv_profit(p_e, p_e, c, alpha, gamma)
    profit_c = sv_profit(p_c, p_c, c, alpha, gamma)

    price_start = 2 * p_e - p_c
    price_end = 2 * p_c - p_e
    prices = np.round(np.linspace(price_start, price_end, m), 3).tolist()

    return prices, p_e, p_c, profit_e, profit_c


# ── Q-learning infrastructure ──────────────────────────────────────

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

def train(discount_factor, seed, alpha, gamma_demand):
    prices, p_e, p_c, profit_e, profit_c = sv_prices_and_benchmarks(
        alpha, gamma_demand, C, M)
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
        eps = epsilon_at(step_count, BETA_EXPLORE)
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
        pi1 = sv_profit(p1_next, p2_next, C, alpha, gamma_demand)
        pi2 = sv_profit(p2_next, p1_next, C, alpha, gamma_demand)
        Q1[s, a1] = (1 - ALPHA_LR) * Q1[s, a1] + ALPHA_LR * (
            pi1 + discount_factor * np.max(Q1[s_next]))
        Q2[s, a2] = (1 - ALPHA_LR) * Q2[s, a2] + ALPHA_LR * (
            pi2 + discount_factor * np.max(Q2[s_next]))
        s = s_next
        if step_count % CHECK_EVERY == 0:
            current_pi1 = greedy_map(Q1)
            current_pi2 = greedy_map(Q2)
            if (np.array_equal(current_pi1, prev_pi1) and
                    np.array_equal(current_pi2, prev_pi2)):
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
        "gamma_demand", "discount_factor", "seed", "converged", "steps",
        "p_e", "p_c", "profit_e", "profit_c", "collusion_premium_pct",
        "cycle_prices", "cycle_length",
        "avg_price_1", "avg_price_2", "avg_profit_1", "avg_profit_2",
        "delta_1", "delta_2",
    ]

    csv_file = open(OUT_FILE, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()
    csv_file.flush()

    total = len(GAMMAS_DEMAND) * RUNS_PER_GAMMA
    count = 0
    t_global = time.time()

    for gamma_d in GAMMAS_DEMAND:
        prices, p_e, p_c, profit_e, profit_c = sv_prices_and_benchmarks(
            ALPHA_DEMAND, gamma_d, C, M)
        premium = 100 * (profit_c - profit_e) / profit_e if profit_e > 0 else 0
        a_val = ALPHA_DEMAND * (1 - gamma_d)
        log(f"\n--- gamma_demand={gamma_d}, a={a_val:.1f}, p_e={p_e:.3f}, "
            f"p_c={p_c:.3f}, pi_e={profit_e:.2f}, pi_c={profit_c:.2f}, "
            f"premium={premium:.0f}% ---")
        log(f"    Grid: [{prices[0]:.3f}, {prices[-1]:.3f}], spacing={prices[1]-prices[0]:.3f}")

        for run in range(RUNS_PER_GAMMA):
            count += 1
            seed = int.from_bytes(os.urandom(3), "big") % 1_000_000
            t0 = time.time()
            Q1, Q2, prices_t, converged, steps = train(
                DISCOUNT_FACTOR, seed, ALPHA_DEMAND, gamma_d)
            elapsed = time.time() - t0

            # Find cycle from middle of grid
            mid = len(prices_t) // 2
            traj, loop_start = simulate(Q1, Q2, prices_t[mid], prices_t[mid], prices_t)

            if loop_start >= 0:
                cycle = traj[loop_start:]
                cycle_str = " -> ".join(f"({p1},{p2})" for p1, p2 in cycle)
                avg_p1 = np.mean([p1 for p1, p2 in cycle])
                avg_p2 = np.mean([p2 for p1, p2 in cycle])
                avg_pi1 = np.mean([sv_profit(p1, p2, C, ALPHA_DEMAND, gamma_d)
                                   for p1, p2 in cycle])
                avg_pi2 = np.mean([sv_profit(p2, p1, C, ALPHA_DEMAND, gamma_d)
                                   for p1, p2 in cycle])
                denom = profit_c - profit_e
                delta1 = (avg_pi1 - profit_e) / denom if denom != 0 else 0
                delta2 = (avg_pi2 - profit_e) / denom if denom != 0 else 0
            else:
                cycle_str = "no cycle"
                avg_p1 = avg_p2 = avg_pi1 = avg_pi2 = delta1 = delta2 = float('nan')
                cycle = []

            log(f"  [{count}/{total}] gamma_d={gamma_d}, run={run+1}, seed={seed}, "
                f"conv={'Y' if converged else 'N'}, steps={steps:,}, "
                f"delta=({delta1:.2f},{delta2:.2f}), {elapsed:.0f}s")

            row = {
                "gamma_demand": gamma_d,
                "discount_factor": DISCOUNT_FACTOR,
                "seed": seed,
                "converged": converged,
                "steps": steps,
                "p_e": round(p_e, 3),
                "p_c": round(p_c, 3),
                "profit_e": round(profit_e, 2),
                "profit_c": round(profit_c, 2),
                "collusion_premium_pct": round(premium, 1),
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
    log("SINGH-VIVES SUBSTITUTABILITY ANALYSIS")
    log(f"alpha={ALPHA_DEMAND}, c={C}, discount_factor={DISCOUNT_FACTOR}")
    log(f"{'='*80}")
    log(f"\n{'gamma':>6} | {'prem%':>6} | {'p_e':>6} | {'Avg D1':>7} | {'Avg D2':>7} | "
        f"{'Std':>6} | {'Conv':>4} | {'Avg len':>7}")
    log("-" * 72)
    for gd in GAMMAS_DEMAND:
        sub = df[df["gamma_demand"] == gd]
        if sub.empty:
            continue
        d1 = pd.to_numeric(sub["delta_1"], errors="coerce")
        d2 = pd.to_numeric(sub["delta_2"], errors="coerce")
        prem = sub["collusion_premium_pct"].iloc[0]
        conv = sub["converged"].sum()
        avg_len = pd.to_numeric(sub["cycle_length"], errors="coerce").mean()
        p_e_val = sub["p_e"].iloc[0]
        log(f"{gd:>6.2f} | {prem:>5.0f}% | {p_e_val:>6.2f} | {d1.mean():>7.2f} | "
            f"{d2.mean():>7.2f} | {d1.std():>6.2f} | {conv:>2}/{len(sub)} | {avg_len:>7.1f}")


if __name__ == "__main__":
    main()
