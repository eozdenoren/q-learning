"""
Calculate net discounted gains from deviation.

For each trained model with a symmetric fixed point:
- Compute the discounted profit stream from staying at cycle forever
- For each possible deviation price, compute the discounted profit stream
  along the punishment path until return to cycle, then cycle forever after
- Net gain = deviation stream - stay stream

If net gain < 0 for all deviations, the cycle is self-enforcing.
"""

import numpy as np
import csv
import os
import time

# ── Parameters ──────────────────────────────────────────────────────
GAMMAS = [0.95, 0.8]
RUNS_PER_GAMMA = 20  # More runs to get good coverage
ALPHA = 0.15
BETA = 4e-6
K1 = 9.0
K2 = 0.67
C = 3.0
M = 15
CHECK_EVERY = 1000
STABLE_REQUIRED = 100_000
MAX_PERIODS = 5_000_000

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deviation_gains.csv")

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


# ── Simulation with full profit tracking ────────────────────────────

def simulate_with_profits(Q1, Q2, start_p1, start_p2, prices, max_steps=200):
    """Follow greedy policy, return list of (p1, p2, pi1, pi2) until cycle."""
    rng_sim = np.random.default_rng(0)
    s = state_index(start_p1, start_p2, prices)
    first_seen = {}
    trajectory = []

    for t in range(max_steps):
        if s in first_seen:
            loop_start = first_seen[s]
            return trajectory, loop_start
        p1, p2 = index_to_state(s, prices)
        pi1 = profit1(p1, p2, C, K1, K2)
        pi2 = profit2(p1, p2, C, K1, K2)
        first_seen[s] = t
        trajectory.append((p1, p2, pi1, pi2))
        a1 = argmax_tie(Q1[s], rng_sim)
        a2 = argmax_tie(Q2[s], rng_sim)
        s = state_index(prices[a1], prices[a2], prices)

    return trajectory, -1


def format_prices(traj):
    return " → ".join(f"({p1},{p2})" for p1, p2, _, _ in traj)


# ── Main ────────────────────────────────────────────────────────────

def main():
    prices, p_e, p_c, profit_e, profit_c = calculate_prices(K1, K2, C, M)
    log(f"p_e={p_e:.3f}, p_c={p_c:.3f}, profit_e={profit_e:.2f}, profit_c={profit_c:.2f}")

    columns = [
        "gamma", "seed", "cycle_price", "cycle_profit",
        "deviation_price", "deviation_1shot_profit",
        "punishment_path", "punishment_length",
        "profits_during_punishment",
        "discounted_deviation_payoff", "discounted_stay_payoff",
        "net_gain", "net_gain_pct",
    ]

    csv_file = open(OUT_FILE, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()
    csv_file.flush()

    count = 0
    t_global = time.time()

    for gamma in GAMMAS:
        for run in range(RUNS_PER_GAMMA):
            seed = int.from_bytes(os.urandom(3), "big") % 1_000_000
            Q1, Q2, prices_t, converged, steps = train(gamma, seed)
            if not converged:
                continue

            # Find symmetric fixed point
            found_fp = False
            cycle_price = None
            for sp in prices_t:
                traj, loop_start = simulate_with_profits(Q1, Q2, sp, sp, prices_t)
                if loop_start >= 0:
                    cycle = traj[loop_start:]
                    if len(cycle) == 1 and cycle[0][0] == cycle[0][1]:
                        cycle_price = cycle[0][0]
                        found_fp = True
                        break
            if not found_fp:
                continue

            cycle_pi = profit1(cycle_price, cycle_price, C, K1, K2)
            cycle_idx = prices_t.index(cycle_price)

            # Discounted payoff from staying at cycle forever: pi / (1 - gamma)
            stay_payoff = cycle_pi / (1 - gamma)

            log(f"gamma={gamma}, run={run+1}, seed={seed}, cycle=({cycle_price},{cycle_price}), "
                f"pi_cycle={cycle_pi:.2f}, V_stay={stay_payoff:.1f}")

            for dev_idx in range(len(prices_t)):
                if dev_idx == cycle_idx:
                    continue  # skip staying at cycle price

                dev_price = prices_t[dev_idx]

                # Simulate: firm 1 deviates, firm 2 at cycle price
                traj, loop_start = simulate_with_profits(
                    Q1, Q2, dev_price, cycle_price, prices_t)

                if loop_start < 0:
                    continue

                # Find when/if we return to cycle
                cycle_return = -1
                for i, (p1, p2, pi1, pi2) in enumerate(traj):
                    if i > 0 and p1 == cycle_price and p2 == cycle_price:
                        cycle_return = i
                        break

                if cycle_return < 0:
                    # Doesn't return to cycle — compute payoff over the cycle found
                    cycle_part = traj[loop_start:]
                    # Discounted payoff of the transient + repeating cycle
                    dev_payoff = 0.0
                    for i, (p1, p2, pi1, pi2) in enumerate(traj[:loop_start]):
                        dev_payoff += (gamma ** i) * pi1
                    # Then the cycle repeats forever
                    cycle_per_period = sum(pi1 for _, _, pi1, _ in cycle_part) / len(cycle_part)
                    # Discounted value of cycle starting at period loop_start
                    dev_payoff += (gamma ** loop_start) * cycle_per_period / (1 - gamma)
                    punishment_len = len(traj[:loop_start])
                    punishment_path = format_prices(traj[:min(15, len(traj))])
                    profits_str = ", ".join(f"{pi1:.1f}" for _, _, pi1, _ in traj[:min(15, len(traj))])
                else:
                    # Returns to cycle — compute payoff of transient + cycle forever
                    dev_payoff = 0.0
                    for i in range(cycle_return):
                        dev_payoff += (gamma ** i) * traj[i][2]  # pi1
                    # From cycle_return onward: cycle_pi forever
                    dev_payoff += (gamma ** cycle_return) * cycle_pi / (1 - gamma)
                    punishment_len = cycle_return
                    punishment_path = format_prices(traj[:cycle_return + 1])
                    profits_str = ", ".join(f"{traj[i][2]:.1f}" for i in range(min(cycle_return + 1, 15)))

                net_gain = dev_payoff - stay_payoff
                net_gain_pct = 100 * net_gain / stay_payoff if stay_payoff != 0 else 0

                count += 1
                row = {
                    "gamma": gamma,
                    "seed": seed,
                    "cycle_price": cycle_price,
                    "cycle_profit": round(cycle_pi, 2),
                    "deviation_price": dev_price,
                    "deviation_1shot_profit": round(profit1(dev_price, cycle_price, C, K1, K2), 2),
                    "punishment_path": punishment_path,
                    "punishment_length": punishment_len,
                    "profits_during_punishment": profits_str,
                    "discounted_deviation_payoff": round(dev_payoff, 2),
                    "discounted_stay_payoff": round(stay_payoff, 2),
                    "net_gain": round(net_gain, 2),
                    "net_gain_pct": round(net_gain_pct, 2),
                }
                writer.writerow(row)
                csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t_global
    log(f"\nDone! {count} results in {elapsed:.0f}s → {OUT_FILE}")

    # Summary
    if count > 0:
        import pandas as pd
        df = pd.read_csv(OUT_FILE)

        for gamma in GAMMAS:
            gdf = df[df["gamma"] == gamma]
            if gdf.empty:
                continue
            log(f"\n{'='*80}")
            log(f"GAMMA = {gamma}")
            log(f"{'='*80}")
            log(f"Models tested: {gdf['seed'].nunique()}")
            log(f"All deviations have negative net gain? {(gdf['net_gain'] < 0).all()}")
            log(f"Max net gain: {gdf['net_gain'].max():.2f} ({gdf['net_gain_pct'].max():.2f}%)")
            log(f"Min net gain: {gdf['net_gain'].min():.2f} ({gdf['net_gain_pct'].min():.2f}%)")

            # By deviation direction
            for direction in ['undercut', 'overcut']:
                if direction == 'undercut':
                    sub = gdf[gdf['deviation_price'] < gdf['cycle_price']]
                else:
                    sub = gdf[gdf['deviation_price'] > gdf['cycle_price']]
                if sub.empty:
                    continue
                log(f"\n  {direction.upper()}:")
                log(f"    Count: {len(sub)}")
                log(f"    All negative? {(sub['net_gain'] < 0).all()}")
                log(f"    Avg net gain: {sub['net_gain'].mean():.2f}")
                log(f"    Avg punishment length: {sub['punishment_length'].mean():.1f}")
                log(f"    Max net gain (closest to profitable): {sub['net_gain'].max():.2f}")

            # Show the most tempting deviations (highest net gain)
            log(f"\n  Most tempting deviations (highest net gain):")
            top = gdf.nlargest(5, 'net_gain')
            for _, row in top.iterrows():
                log(f"    dev={row['deviation_price']}, 1shot_pi={row['deviation_1shot_profit']:.1f}, "
                    f"net={row['net_gain']:.2f} ({row['net_gain_pct']:.1f}%), "
                    f"punishment={row['punishment_length']} steps")
                log(f"      profits: {row['profits_during_punishment']}")


if __name__ == "__main__":
    main()
