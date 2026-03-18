"""
Collusion sustainability experiment for Exercise 3.

For gamma=0.95 (and 0.8 for comparison), trains models and tests:
1. Starting from converged prices (confirm stability)
2. Starting from (deviation_price, cycle_price) — does punishment + forgiveness occur?
3. Starting from (p_c, p_c) — can full collusion be sustained?
4. Net discounted gains from deviation — is the cycle self-enforcing?

Writes results incrementally to collusion_sustainability.csv.
"""

import numpy as np
import csv
import os
import time

# ── Parameters ──────────────────────────────────────────────────────
GAMMAS = [0.95, 0.8]
RUNS_PER_GAMMA = 15
ALPHA = 0.15
BETA = 4e-6
K1 = 9.0
K2 = 0.67
C = 3.0
M = 15
CHECK_EVERY = 1000
STABLE_REQUIRED = 100_000
MAX_PERIODS = 5_000_000

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collusion_sustainability.csv")

# ── Core functions ──────────────────────────────────────────────────

def demand1(p1, p2, k1, k2):
    return max(0.0, k1 - p1 + k2 * p2)

def profit1(p1, p2, c, k1, k2):
    return (p1 - c) * demand1(p1, p2, k1, k2)

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
        pi2 = (p2_next - C) * max(0.0, K1 - p2_next + K2 * p1_next)
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

def format_traj(traj, max_show=15):
    return " -> ".join(f"({p1},{p2})" for p1, p2 in traj[:max_show])


# ── Main ────────────────────────────────────────────────────────────

def main():
    prices_base, p_e, p_c, profit_e, profit_c = calculate_prices(K1, K2, C, M)
    log(f"p_e={p_e:.3f}, p_c={p_c:.3f}, profit_e={profit_e:.2f}, profit_c={profit_c:.2f}")
    log(f"Prices: {prices_base}")

    # Find closest prices in grid to p_e and p_c
    p_e_grid = min(prices_base, key=lambda p: abs(p - p_e))
    p_c_grid = min(prices_base, key=lambda p: abs(p - p_c))
    log(f"Closest to p_e: {p_e_grid}, closest to p_c: {p_c_grid}")

    columns = [
        "gamma", "seed", "steps",
        "cycle_price", "cycle_profit", "delta_at_cycle",
        "test_type", "start_p1", "start_p2",
        "trajectory", "returns_to_cycle", "steps_to_return",
        "end_cycle_prices", "end_cycle_length",
        "net_deviation_gain", "net_deviation_gain_pct",
    ]

    csv_file = open(OUT_FILE, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()
    csv_file.flush()

    count = 0
    t_global = time.time()
    total = len(GAMMAS) * RUNS_PER_GAMMA

    for gamma in GAMMAS:
        for run in range(RUNS_PER_GAMMA):
            seed = int.from_bytes(os.urandom(3), "big") % 1_000_000
            t0 = time.time()
            Q1, Q2, prices_t, converged, steps = train(gamma, seed)
            train_time = time.time() - t0

            if not converged:
                log(f"  gamma={gamma}, run={run+1}, seed={seed}: not converged, skipping")
                continue

            # Find symmetric fixed point
            cycle_price = None
            for sp in prices_t:
                traj, loop_start = simulate(Q1, Q2, sp, sp, prices_t)
                if loop_start >= 0:
                    cycle = traj[loop_start:]
                    if len(cycle) == 1 and cycle[0][0] == cycle[0][1]:
                        cycle_price = cycle[0][0]
                        break

            if cycle_price is None:
                log(f"  gamma={gamma}, run={run+1}, seed={seed}: no symmetric fixed point, skipping")
                continue

            cycle_pi = profit1(cycle_price, cycle_price, C, K1, K2)
            delta_cycle = (cycle_pi - profit_e) / (profit_c - profit_e) if profit_c != profit_e else 0
            stay_payoff = cycle_pi / (1 - gamma)

            run_idx = run + 1
            log(f"\ngamma={gamma}, run={run_idx}, seed={seed}, cycle=({cycle_price},{cycle_price}), "
                f"delta={delta_cycle:.2f}, {train_time:.0f}s")

            # ── Test 1: Stability — start from cycle ──
            traj, ls = simulate(Q1, Q2, cycle_price, cycle_price, prices_t)
            row = {
                "gamma": gamma, "seed": seed, "steps": steps,
                "cycle_price": cycle_price, "cycle_profit": round(cycle_pi, 2),
                "delta_at_cycle": round(delta_cycle, 3),
                "test_type": "stability",
                "start_p1": cycle_price, "start_p2": cycle_price,
                "trajectory": format_traj(traj),
                "returns_to_cycle": True,
                "steps_to_return": 0,
                "end_cycle_prices": f"({cycle_price},{cycle_price})",
                "end_cycle_length": 1,
                "net_deviation_gain": 0, "net_deviation_gain_pct": 0,
            }
            writer.writerow(row)
            count += 1

            # ── Test 2: Full collusion — start from (p_c, p_c) ──
            traj, ls = simulate(Q1, Q2, p_c_grid, p_c_grid, prices_t)
            returns = False
            steps_to_return = -1
            for i, (p1, p2) in enumerate(traj):
                if i > 0 and p1 == cycle_price and p2 == cycle_price:
                    returns = True
                    steps_to_return = i
                    break
            end_cycle = traj[ls:] if ls >= 0 else []
            row = {
                "gamma": gamma, "seed": seed, "steps": steps,
                "cycle_price": cycle_price, "cycle_profit": round(cycle_pi, 2),
                "delta_at_cycle": round(delta_cycle, 3),
                "test_type": "full_collusion",
                "start_p1": p_c_grid, "start_p2": p_c_grid,
                "trajectory": format_traj(traj),
                "returns_to_cycle": returns,
                "steps_to_return": steps_to_return if returns else len(traj),
                "end_cycle_prices": " -> ".join(f"({p1},{p2})" for p1, p2 in end_cycle[:5]),
                "end_cycle_length": len(end_cycle),
                "net_deviation_gain": "", "net_deviation_gain_pct": "",
            }
            writer.writerow(row)
            count += 1
            log(f"  Full collusion ({p_c_grid},{p_c_grid}): {'returns' if returns else 'does NOT return'} "
                f"to cycle in {steps_to_return if returns else '?'} steps")
            log(f"    Path: {format_traj(traj, 10)}")

            # ── Test 3: Deviations — firm 1 undercuts from cycle ──
            cycle_idx = prices_t.index(cycle_price)
            for dev_idx in range(len(prices_t)):
                if dev_idx == cycle_idx:
                    continue
                dev_price = prices_t[dev_idx]
                traj, ls = simulate(Q1, Q2, dev_price, cycle_price, prices_t)

                returns = False
                steps_to_return = -1
                for i, (p1, p2) in enumerate(traj):
                    if i > 0 and p1 == cycle_price and p2 == cycle_price:
                        returns = True
                        steps_to_return = i
                        break

                # Compute net deviation gain
                if returns:
                    dev_payoff = 0.0
                    for i in range(steps_to_return):
                        pi_dev = profit1(traj[i][0], traj[i][1], C, K1, K2)
                        dev_payoff += (gamma ** i) * pi_dev
                    dev_payoff += (gamma ** steps_to_return) * stay_payoff
                    net_gain = dev_payoff - stay_payoff
                    net_gain_pct = 100 * net_gain / stay_payoff if stay_payoff != 0 else 0
                else:
                    net_gain = ""
                    net_gain_pct = ""

                direction = "undercut" if dev_price < cycle_price else "overcut"
                end_cycle = traj[ls:] if ls >= 0 else []
                row = {
                    "gamma": gamma, "seed": seed, "steps": steps,
                    "cycle_price": cycle_price, "cycle_profit": round(cycle_pi, 2),
                    "delta_at_cycle": round(delta_cycle, 3),
                    "test_type": f"deviation_{direction}",
                    "start_p1": dev_price, "start_p2": cycle_price,
                    "trajectory": format_traj(traj),
                    "returns_to_cycle": returns,
                    "steps_to_return": steps_to_return if returns else len(traj),
                    "end_cycle_prices": " -> ".join(f"({p1},{p2})" for p1, p2 in end_cycle[:5]),
                    "end_cycle_length": len(end_cycle),
                    "net_deviation_gain": round(net_gain, 2) if isinstance(net_gain, float) else "",
                    "net_deviation_gain_pct": round(net_gain_pct, 2) if isinstance(net_gain_pct, float) else "",
                }
                writer.writerow(row)
                count += 1

            csv_file.flush()

    csv_file.close()
    total_time = time.time() - t_global
    log(f"\nDone! {count} rows in {total_time:.0f}s -> {OUT_FILE}")

    # Summary
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

        # Stability
        stab = gdf[gdf["test_type"] == "stability"]
        log(f"\nStability: {len(stab)} tests, all return to cycle: {stab['returns_to_cycle'].all()}")

        # Full collusion
        fc = gdf[gdf["test_type"] == "full_collusion"]
        log(f"\nFull collusion tests: {len(fc)}")
        log(f"  Returns to partial cycle: {fc['returns_to_cycle'].sum()}/{len(fc)}")
        log(f"  Avg steps to return: {fc.loc[fc['returns_to_cycle'], 'steps_to_return'].mean():.1f}")

        # Deviations
        for direction in ["undercut", "overcut"]:
            dev = gdf[gdf["test_type"] == f"deviation_{direction}"]
            if dev.empty:
                continue
            log(f"\n{direction.upper()} deviations: {len(dev)}")
            log(f"  Returns to cycle: {dev['returns_to_cycle'].sum()}/{len(dev)} "
                f"({100*dev['returns_to_cycle'].mean():.0f}%)")
            ret = dev[dev["returns_to_cycle"]]
            if not ret.empty:
                log(f"  Avg punishment length: {ret['steps_to_return'].mean():.1f} steps")
                gains = pd.to_numeric(ret["net_deviation_gain"], errors="coerce")
                log(f"  All deviations unprofitable? {(gains < 0).all()}")
                log(f"  Max net gain: {gains.max():.2f}")
                log(f"  Avg net gain: {gains.mean():.2f}")


if __name__ == "__main__":
    main()
