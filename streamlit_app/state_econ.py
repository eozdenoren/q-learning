"""Session state management for Economics Pricing Q-learning demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from numba import njit

# Performance: Limit checkpoint history to prevent memory bloat
MAX_CHECKPOINTS = 50

__all__ = [
    "init_session_state_econ",
    "pick_random_starting_prices_econ",
    "step_agent_econ",
    "run_batch_training_econ",
    "run_until_convergence_econ",
    "get_display_state_econ",
    "is_in_playback_mode_econ",
    "save_checkpoint_econ",
    "rewind_checkpoint_econ",
    "forward_checkpoint_econ",
    "jump_to_latest_econ",
    "jump_to_start_econ",
    "flip_q_table_states",
    "run_single_experiment",
]


# ---------- Price label formatting ----------

def _price_fmt(prices: list) -> str:
    """Return a format string with enough decimals to keep all price labels unique."""
    for d in range(1, 5):
        labels = [f"{p:.{d}f}" for p in prices]
        if len(set(labels)) == len(labels):
            return f".{d}f"
    return ".3f"


# ---------- Economics Helper Functions ----------


def demand1(p1: float, p2: float, t: float, v: float) -> float:
    """Hotelling demand for firm 1 with uncovered market.

    Accounts for consumers who choose not to buy if price is too high.
    q1 = min(x_hat, (v - p1)/t) where x_hat = 1/2 + (p2 - p1)/(2t).
    """
    if p1 >= v:
        return 0.0
    x_hat = 0.5 + (p2 - p1) / (2 * t)
    x_max = (v - p1) / t
    return max(0.0, min(1.0, min(x_hat, x_max)))


def demand2(p1: float, p2: float, t: float, v: float) -> float:
    """Hotelling demand for firm 2 with uncovered market.

    q2 = min(1 - x_hat, (v - p2)/t) where x_hat = 1/2 + (p2 - p1)/(2t).
    """
    if p2 >= v:
        return 0.0
    x_hat = 0.5 + (p2 - p1) / (2 * t)
    x_max = (v - p2) / t
    return max(0.0, min(1.0, min(1.0 - x_hat, x_max)))


def profit1(p1: float, p2: float, c: float, t: float, v: float) -> float:
    """Profit for firm 1: π1 = (p1 - c) * q1."""
    return (p1 - c) * demand1(p1, p2, t, v)


def profit2(p1: float, p2: float, c: float, t: float, v: float) -> float:
    """Profit for firm 2: π2 = (p2 - c) * q2."""
    return (p2 - c) * demand2(p1, p2, t, v)


def calculate_prices(
    t: float, v: float, c: float, m: int
) -> tuple[list[float], float, float, float, float]:
    """Calculate equilibrium price, collusion price, and action space (Hotelling model).

    Returns:
        (prices list, p_e (Nash equilibrium), p_c (collusion), profit_e, profit_c)
    """

    # Nash equilibrium price: p_e = c + t
    p_e = c + t

    # Nash equilibrium profit per firm: π_e = (p_e - c) * 1/2 = t/2
    profit_e = t / 2.0

    # Collusion price: p_c = v - t/2  (constrained by consumer willingness to pay)
    p_c = v - t / 2.0

    # Collusion profit per firm: π_c = (p_c - c) * 1/2
    profit_c = (p_c - c) / 2.0

    # Feasible price interval: [2p_e - p_c, 2p_c - p_e]
    # m equally spaced prices, following Calvano et al. (2020)
    price_start = 2 * p_e - p_c
    price_end = 2 * p_c - p_e

    # Guard: ensure grid is valid (p_c > p_e) and non-negative
    if price_start >= price_end:
        price_start, price_end = price_end, price_start
    price_start = max(0.0, price_start)
    if price_end <= price_start:
        price_end = price_start + 1.0

    price = np.round(np.linspace(price_start, price_end, m), 3).tolist()

    return price, p_e, p_c, profit_e, profit_c


def state_index(p1: float, p2: float, prices: list[float]) -> int:
    """Flattens a 2D (p1, p2) into a single row index."""
    price_to_idx = {p: i for i, p in enumerate(prices)}
    n_actions = len(prices)
    return price_to_idx[p1] * n_actions + price_to_idx[p2]


def index_to_state(s: int, prices: list[float]) -> tuple[float, float]:
    """Inverse map from index -> (p1, p2)."""
    n_actions = len(prices)
    i = s // n_actions
    j = s % n_actions
    return prices[i], prices[j]


def flip_q_table_states(Q: np.ndarray, prices: list[float]) -> np.ndarray:
    """Flip state indices in a Q-table from s(p1, p2) to s(p2, p1).

    This function is used when a Q-table trained from one player's perspective
    needs to be used from another player's perspective. It transforms all states
    by swapping p1 and p2 in the state encoding.

    Args:
        Q: Q-table array of shape (n_states, n_actions) where n_states = len(prices)^2
        prices: List of price values used to encode states

    Returns:
        A new Q-table with flipped state indices. Q_flipped[s_flipped] = Q[s]
        where s_flipped is the state index for s(p2, p1) when s was for s(p1, p2).
    """
    n_states = Q.shape[0]
    Q_flipped = np.zeros_like(Q)

    for s in range(n_states):
        p1, p2 = index_to_state(s, prices)
        # Intentionally swap p1 and p2 to flip state perspective
        s_flipped = state_index(p1=p2, p2=p1, prices=prices)
        Q_flipped[s_flipped] = Q[s]  # Copy Q-values to flipped state

    return Q_flipped


def epsilon_at(step: int, beta: float) -> float:
    """Exponential decay exploration rate: ε_t = exp(-beta * t)."""
    return float(np.exp(-beta * step))


def argmax_tie(x: np.ndarray, rng: np.random.Generator) -> int:
    """When there are multiple max values, choose a random one."""
    m = np.max(x)
    idxs = np.flatnonzero(np.isclose(x, m))
    return int(rng.choice(idxs))


def greedy_map(Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Returns an array of length n_states with best-action indices.

    Uses np.argmax (deterministic) so convergence checks don't consume
    the training RNG. All-zero rows always return action 0.
    """
    return np.argmax(Q, axis=1)


# ---------- Numba JIT Core Loop ----------


@njit(cache=True)
def _train_loop_jit(prices, n, c, t, v, alpha, gamma, beta,
                    Q1, Q2, s_init, check_every, stable_required,
                    max_periods, rng_seed):
    """Core Q-learning training loop compiled to machine code via Numba.

    Returns (steps_run, final_state, stable_count, converged).
    Q1 and Q2 are modified in-place.
    """
    np.random.seed(rng_seed)
    s = s_init

    prev_pi1 = np.zeros(n * n, dtype=np.int64)
    prev_pi2 = np.zeros(n * n, dtype=np.int64)
    stable = 0
    step = 0

    for step in range(1, max_periods + 1):
        eps = np.exp(-beta * step)

        # Player 1: epsilon-greedy
        if np.random.random() < eps:
            a1 = np.random.randint(0, n)
        else:
            a1 = 0
            mx = Q1[s, 0]
            for j in range(1, n):
                if Q1[s, j] > mx:
                    mx = Q1[s, j]
                    a1 = j

        # Player 2: epsilon-greedy
        if np.random.random() < eps:
            a2 = np.random.randint(0, n)
        else:
            a2 = 0
            mx = Q2[s, 0]
            for j in range(1, n):
                if Q2[s, j] > mx:
                    mx = Q2[s, j]
                    a2 = j

        p1n = prices[a1]
        p2n = prices[a2]
        sn = a1 * n + a2

        # Demand 1
        if p1n >= v:
            d1 = 0.0
        else:
            x_hat = 0.5 + (p2n - p1n) / (2.0 * t)
            x_max = (v - p1n) / t
            d1 = min(x_hat, x_max)
            if d1 < 0.0:
                d1 = 0.0
            elif d1 > 1.0:
                d1 = 1.0

        # Demand 2
        if p2n >= v:
            d2 = 0.0
        else:
            x_hat2 = 0.5 + (p1n - p2n) / (2.0 * t)
            x_max2 = (v - p2n) / t
            d2 = min(x_hat2, x_max2)
            if d2 < 0.0:
                d2 = 0.0
            elif d2 > 1.0:
                d2 = 1.0

        pi1 = (p1n - c) * d1
        pi2 = (p2n - c) * d2

        # Max Q for next state
        max_q1 = Q1[sn, 0]
        max_q2 = Q2[sn, 0]
        for j in range(1, n):
            if Q1[sn, j] > max_q1:
                max_q1 = Q1[sn, j]
            if Q2[sn, j] > max_q2:
                max_q2 = Q2[sn, j]

        # Q-learning updates
        Q1[s, a1] = (1.0 - alpha) * Q1[s, a1] + alpha * (pi1 + gamma * max_q1)
        Q2[s, a2] = (1.0 - alpha) * Q2[s, a2] + alpha * (pi2 + gamma * max_q2)
        s = sn

        # Convergence check
        if step % check_every == 0:
            changed = False
            for i in range(n * n):
                best1 = 0
                mx1 = Q1[i, 0]
                for j in range(1, n):
                    if Q1[i, j] > mx1:
                        mx1 = Q1[i, j]
                        best1 = j
                if best1 != prev_pi1[i]:
                    changed = True
                prev_pi1[i] = best1

                best2 = 0
                mx2 = Q2[i, 0]
                for j in range(1, n):
                    if Q2[i, j] > mx2:
                        mx2 = Q2[i, j]
                        best2 = j
                if best2 != prev_pi2[i]:
                    changed = True
                prev_pi2[i] = best2

            if not changed:
                stable += check_every
                if stable >= stable_required:
                    return step, s, stable, True
            else:
                stable = 0

    return step, s, stable, False


# ---------- Standalone Experiment (no session state) ----------


def run_single_experiment(
    c: float, t: float, v: float, m: int,
    alpha: float, gamma: float, beta: float, seed: int,
    check_every: int = 1000, stable_required: int = 100000,
    max_periods: int = 2000000,
) -> dict:
    """Run a full Q-learning experiment without touching Streamlit session state.

    Returns a dict with converged prices, Delta values, cycle info, etc.
    """
    prices, p_e, p_c, profit_e, profit_c = calculate_prices(t, v, c, m)
    n = len(prices)
    n_states = n * n
    prices_arr = np.array(prices, dtype=np.float64)

    rng = np.random.default_rng(seed)
    Q1 = np.zeros((n_states, n), dtype=np.float64)
    Q2 = np.zeros((n_states, n), dtype=np.float64)

    # Random starting state
    a1_start = int(rng.integers(0, n))
    a2_start = int(rng.integers(0, n))
    s = a1_start * n + a2_start
    start_p1, start_p2 = prices[a1_start], prices[a2_start]

    # Run JIT-compiled training loop
    step, s, _stable, converged = _train_loop_jit(
        prices_arr, n, c, t, v, alpha, gamma, beta,
        Q1, Q2, s, check_every, stable_required, max_periods, seed,
    )

    # Follow greedy policy to find equilibrium cycle
    s_eq = s
    for _ in range(200):
        a1 = int(np.argmax(Q1[s_eq]))
        a2 = int(np.argmax(Q2[s_eq]))
        s_eq = a1 * n + a2

    # Detect cycle
    visited = {}
    path = []
    for i in range(1000):
        if s_eq in visited:
            loop_start = visited[s_eq]
            cycle = path[loop_start:]
            break
        visited[s_eq] = i
        a1 = int(np.argmax(Q1[s_eq]))
        a2 = int(np.argmax(Q2[s_eq]))
        p1_act, p2_act = prices[a1], prices[a2]
        path.append((p1_act, p2_act))
        s_eq = a1 * n + a2
    else:
        cycle = [path[-1]] if path else [(prices[0], prices[0])]

    # Compute average prices and profits over cycle
    avg_p1 = np.mean([p[0] for p in cycle])
    avg_p2 = np.mean([p[1] for p in cycle])
    avg_pi1 = np.mean([profit1(p[0], p[1], c, t, v) for p in cycle])
    avg_pi2 = np.mean([profit2(p[0], p[1], c, t, v) for p in cycle])

    # Delta
    denom = profit_c - profit_e
    d1 = (avg_pi1 - profit_e) / denom if abs(denom) > 1e-10 else 0.0
    d2 = (avg_pi2 - profit_e) / denom if abs(denom) > 1e-10 else 0.0

    # Cycle string
    cycle_str = " → ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in cycle)

    return {
        "converged": converged,
        "steps": step,
        "seed": seed,
        "start_p1": start_p1,
        "start_p2": start_p2,
        "cycle_len": len(cycle),
        "cycle_str": cycle_str,
        "avg_p1": avg_p1,
        "avg_p2": avg_p2,
        "avg_pi1": avg_pi1,
        "avg_pi2": avg_pi2,
        "delta1": d1,
        "delta2": d2,
        "p_e": p_e,
        "p_c": p_c,
        "profit_e": profit_e,
        "profit_c": profit_c,
    }


# ---------- State Management Functions ----------


def init_session_state_econ(config: dict) -> None:
    """Initialize or reset all session state for economics pricing (tab-scoped)."""
    tab_id = config.get("tab_id", "default")
    t = config["t"]
    v = config["v"]
    c = config["c"]
    m = config["m"]
    alpha = config["alpha"]
    delta = config["delta"]
    beta = config["beta"]
    seed = config.get("seed", 43)

    # Calculate prices and key prices
    prices, p_e, p_c, profit_e, profit_c = calculate_prices(t, v, c, m)
    n_actions = len(prices)
    n_states = n_actions * n_actions

    # Store configuration
    st.session_state[f"{tab_id}_prices"] = prices
    st.session_state[f"{tab_id}_n_actions"] = n_actions
    st.session_state[f"{tab_id}_n_states"] = n_states
    # Store config values under cfg_ keys to avoid widget key collisions
    st.session_state[f"{tab_id}_cfg_t"] = t
    st.session_state[f"{tab_id}_cfg_v"] = v
    st.session_state[f"{tab_id}_cfg_c"] = c
    st.session_state[f"{tab_id}_cfg_m"] = m
    st.session_state[f"{tab_id}_cfg_alpha"] = alpha
    st.session_state[f"{tab_id}_cfg_delta"] = delta
    st.session_state[f"{tab_id}_cfg_beta"] = beta
    st.session_state[f"{tab_id}_cfg_p_e"] = p_e
    st.session_state[f"{tab_id}_cfg_p_c"] = p_c
    st.session_state[f"{tab_id}_cfg_profit_e"] = profit_e
    st.session_state[f"{tab_id}_cfg_profit_c"] = profit_c
    st.session_state[f"{tab_id}_cfg_seed"] = seed

    # Initialize random number generator
    rng = np.random.default_rng(seed)
    st.session_state[f"{tab_id}_rng"] = rng

    # Initialize Q-tables to zero (standard)
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    st.session_state[f"{tab_id}_Q1"] = Q1
    st.session_state[f"{tab_id}_Q2"] = Q2

    # Create Q-tables as DataFrames for display
    _fmt = _price_fmt(prices)
    states = [f"s({p1:{_fmt}},{p2:{_fmt}})" for p1 in prices for p2 in prices]
    actions = [f"price={p:{_fmt}}" for p in prices]
    st.session_state[f"{tab_id}_q_table_1"] = pd.DataFrame(
        Q1, index=states, columns=actions
    )
    st.session_state[f"{tab_id}_q_table_2"] = pd.DataFrame(
        Q2, index=states, columns=actions
    )

    # Pick starting state based on mode
    start_mode = config.get("start_mode", "Randomised")
    if start_mode == "Fixed":
        p1_start = config.get("fixed_start_p1", prices[0])
        p2_start = config.get("fixed_start_p2", prices[0])
        # Validate prices are in prices (round to nearest if close)
        p1_start = min(prices, key=lambda x: abs(x - p1_start))
        p2_start = min(prices, key=lambda x: abs(x - p2_start))
        starting_prices_picked = True
    else:
        # Randomised: don't set starting prices yet - user must click button
        p1_start = None
        p2_start = None
        starting_prices_picked = False

    # Current state tracking
    if starting_prices_picked:
        s_start = state_index(p1_start, p2_start, prices)
        st.session_state[f"{tab_id}_current_p1"] = p1_start
        st.session_state[f"{tab_id}_current_p2"] = p2_start
        st.session_state[f"{tab_id}_current_state"] = s_start
        # Price history: list of (step_num, p1, p2) tuples
        st.session_state[f"{tab_id}_price_history"] = [(0, p1_start, p2_start)]
    else:
        # No starting prices yet - set to None/empty
        st.session_state[f"{tab_id}_current_p1"] = None
        st.session_state[f"{tab_id}_current_p2"] = None
        st.session_state[f"{tab_id}_current_state"] = None
        st.session_state[f"{tab_id}_price_history"] = []

    # Track whether starting prices have been picked
    st.session_state[f"{tab_id}_starting_prices_picked"] = starting_prices_picked

    # Skipped steps during fast-forward: list of (start_step, end_step) tuples
    st.session_state[f"{tab_id}_skipped_steps"] = []

    # Step counter
    st.session_state[f"{tab_id}_step_count"] = 0

    # Convergence tracking
    st.session_state[f"{tab_id}_prev_pi1"] = greedy_map(Q1, rng)
    st.session_state[f"{tab_id}_prev_pi2"] = greedy_map(Q2, rng)
    st.session_state[f"{tab_id}_stable_count"] = 0
    st.session_state[f"{tab_id}_cfg_check_every"] = config.get("check_every", 1000)
    st.session_state[f"{tab_id}_cfg_stable_required"] = config.get(
        "stable_required", 100000
    )
    st.session_state[f"{tab_id}_cfg_max_periods"] = config.get("max_periods", 2000000)

    # Convergence info
    st.session_state[f"{tab_id}_convergence_info"] = None

    # Clear previous trajectory results
    st.session_state[f"{tab_id}_trajectory"] = None
    st.session_state[f"{tab_id}_trajectory_start"] = None

    # Logging
    st.session_state[f"{tab_id}_step_log"] = []  # Detailed step log
    st.session_state[f"{tab_id}_checkpoints"] = []  # User action checkpoints
    st.session_state[f"{tab_id}_playback_index"] = (
        -1
    )  # -1 = live, 0+ = checkpoint index

    # Ready state
    st.session_state[f"{tab_id}_ready_for_training"] = True

    # Save initial checkpoint
    save_checkpoint_econ(config, "init", {"description": "Initial state"})


def pick_random_starting_prices_econ(config: dict) -> None:
    """Pick random starting prices for randomized mode (tab-scoped)."""
    tab_id = config.get("tab_id", "default")

    # Get prices from session state
    prices = st.session_state.get(f"{tab_id}_prices")
    if not prices:
        return

    # Pick random starting prices using np.random.choice directly (not seeded rng) for true randomization
    p1_start = float(np.random.choice(prices))
    p2_start = float(np.random.choice(prices))

    # Set starting state
    s_start = state_index(p1_start, p2_start, prices)
    st.session_state[f"{tab_id}_current_p1"] = p1_start
    st.session_state[f"{tab_id}_current_p2"] = p2_start
    st.session_state[f"{tab_id}_current_state"] = s_start

    # Add to price history
    st.session_state[f"{tab_id}_price_history"] = [(0, p1_start, p2_start)]

    # Mark as picked
    st.session_state[f"{tab_id}_starting_prices_picked"] = True

    # Save checkpoint
    save_checkpoint_econ(
        config, "pick_starting_prices", {"p1": p1_start, "p2": p2_start}
    )


def step_agent_econ(config: dict) -> None:
    """Perform one Q-learning step for both players with logging (tab-scoped)."""
    tab_id = config.get("tab_id", "default")

    # Check if starting prices have been picked
    starting_prices_picked = st.session_state.get(
        f"{tab_id}_starting_prices_picked", True
    )
    if not starting_prices_picked:
        # Cannot step if starting prices haven't been picked
        return

    # Get configuration
    prices = st.session_state[f"{tab_id}_prices"]
    n_actions = st.session_state[f"{tab_id}_n_actions"]
    t = st.session_state[f"{tab_id}_cfg_t"]
    v = st.session_state[f"{tab_id}_cfg_v"]
    c = st.session_state[f"{tab_id}_cfg_c"]
    alpha = st.session_state[f"{tab_id}_cfg_alpha"]
    delta = st.session_state[f"{tab_id}_cfg_delta"]
    beta = st.session_state[f"{tab_id}_cfg_beta"]
    rng = st.session_state[f"{tab_id}_rng"]

    # Get current state
    s = st.session_state[f"{tab_id}_current_state"]
    p1, p2 = index_to_state(s, prices)

    # Get Q-tables
    Q1 = st.session_state[f"{tab_id}_Q1"]
    Q2 = st.session_state[f"{tab_id}_Q2"]

    # Increment step count
    step_count = st.session_state[f"{tab_id}_step_count"] + 1
    st.session_state[f"{tab_id}_step_count"] = step_count

    # Calculate epsilon for this step
    eps = epsilon_at(step_count, beta)

    # ε-greedy action selection for both players
    if rng.random() < eps:
        a1 = rng.integers(0, n_actions)
        decision_type_1 = "Exploratory (Random)"
    else:
        a1 = argmax_tie(Q1[s], rng)
        decision_type_1 = "Max Value (Greedy)"

    if rng.random() < eps:
        a2 = rng.integers(0, n_actions)
        decision_type_2 = "Exploratory (Random)"
    else:
        a2 = argmax_tie(Q2[s], rng)
        decision_type_2 = "Max Value (Greedy)"

    # Compute next state
    p1_next = prices[a1]
    p2_next = prices[a2]
    s_next = state_index(p1_next, p2_next, prices)

    # Calculate demands and profits (rewards)
    q1 = demand1(p1_next, p2_next, t, v)
    q2 = demand2(p1_next, p2_next, t, v)
    pi1 = profit1(p1_next, p2_next, c, t, v)
    pi2 = profit2(p1_next, p2_next, c, t, v)

    # Q-learning updates
    old_val_1 = Q1[s, a1]
    max_next_q1 = np.max(Q1[s_next])
    td_target_1 = pi1 + delta * max_next_q1
    new_val_1 = (1 - alpha) * old_val_1 + alpha * td_target_1

    old_val_2 = Q2[s, a2]
    max_next_q2 = np.max(Q2[s_next])
    td_target_2 = pi2 + delta * max_next_q2
    new_val_2 = (1 - alpha) * old_val_2 + alpha * td_target_2

    # Update Q-tables
    Q1[s, a1] = new_val_1
    Q2[s, a2] = new_val_2
    st.session_state[f"{tab_id}_Q1"] = Q1
    st.session_state[f"{tab_id}_Q2"] = Q2

    # Update DataFrame representations
    _fmt = _price_fmt(prices)
    states = [f"s({p1:{_fmt}},{p2:{_fmt}})" for p1 in prices for p2 in prices]
    actions = [f"price={p:{_fmt}}" for p in prices]
    st.session_state[f"{tab_id}_q_table_1"] = pd.DataFrame(
        Q1, index=states, columns=actions
    )
    st.session_state[f"{tab_id}_q_table_2"] = pd.DataFrame(
        Q2, index=states, columns=actions
    )

    # Find best actions for next state (for display)
    max_next_a1_idx = argmax_tie(Q1[s_next], rng)
    max_next_a2_idx = argmax_tie(Q2[s_next], rng)
    max_next_p1 = prices[max_next_a1_idx]
    max_next_p2 = prices[max_next_a2_idx]

    # Log step details for Q1
    eq_str_1 = (
        rf"Q_1(s({p1:.1f},{p2:.1f}), a_1={p1_next:.1f} = {old_val_1:.4f} + {alpha} * "
        f"[{pi1:.4f} + {delta} * {max_next_q1:.4f} - {old_val_1:.4f}] = **{new_val_1:.4f}**"
    )

    step_entry_1 = {
        "Player": "Scoopy Doo (Q1)",
        "Step": step_count,
        "State (s)": f"s({p1:.1f},{p2:.1f})",
        "Action (a1)": f"{p1_next:.1f}",
        "Action (a2)": f"{p2_next:.1f}",
        "Next state": f"s({p1_next:.1f},{p2_next:.1f})",
        "Next action (a1)": f"{max_next_p1:.1f}",
        "Next action (a2)": f"{max_next_p2:.1f}",
        "Max next Q1": max_next_q1,
        "Type": decision_type_1,
        "Equation": eq_str_1,
        "New Q": new_val_1,
        "Reward (π1)": pi1,
        "p1": p1_next,
        "p2": p2_next,
        "c": c,
        "q1": q1,
    }

    # Log step details for Q2
    eq_str_2 = (
        rf"Q_2(s({p1:.1f},{p2:.1f}), a_2={p2_next:.1f}) = {old_val_2:.4f} + {alpha} * "
        f"[{pi2:.4f} + {delta} * {max_next_q2:.4f} - {old_val_2:.4f}] = **{new_val_2:.4f}**"
    )

    step_entry_2 = {
        "Player": "Cone Solo (Q2)",
        "Step": step_count,
        "State (s)": f"s({p1:.1f},{p2:.1f})",
        "Action (a1)": f"{p1_next:.1f}",
        "Action (a2)": f"{p2_next:.1f}",
        "Next state": f"s({p1_next:.1f},{p2_next:.1f})",
        "Next action (a1)": f"{max_next_p1:.1f}",
        "Next action (a2)": f"{max_next_p2:.1f}",
        "Max next Q2": max_next_q2,
        "Type": decision_type_2,
        "Equation": eq_str_2,
        "New Q": new_val_2,
        "Reward (π2)": pi2,
        "p1": p1_next,
        "p2": p2_next,
        "c": c,
        "q2": q2,
    }

    st.session_state[f"{tab_id}_step_log"].append(step_entry_1)
    st.session_state[f"{tab_id}_step_log"].append(step_entry_2)

    # Update current state
    st.session_state[f"{tab_id}_current_p1"] = p1_next
    st.session_state[f"{tab_id}_current_p2"] = p2_next
    st.session_state[f"{tab_id}_current_state"] = s_next

    # Update price history (only for step-by-step, not fast-forward)
    st.session_state[f"{tab_id}_price_history"].append((step_count, p1_next, p2_next))

    # Check convergence (every check_every steps)
    check_every = st.session_state[f"{tab_id}_cfg_check_every"]
    if step_count % check_every == 0:
        current_pi1 = greedy_map(Q1, rng)
        current_pi2 = greedy_map(Q2, rng)
        prev_pi1 = st.session_state[f"{tab_id}_prev_pi1"]
        prev_pi2 = st.session_state[f"{tab_id}_prev_pi2"]

        if (np.array_equal(current_pi1, prev_pi1)
                and np.array_equal(current_pi2, prev_pi2)):
            stable_count = st.session_state[f"{tab_id}_stable_count"] + check_every
            st.session_state[f"{tab_id}_stable_count"] = stable_count

            stable_required = st.session_state[f"{tab_id}_cfg_stable_required"]
            if stable_count >= stable_required:
                # Converged!
                st.session_state[f"{tab_id}_convergence_info"] = {
                    "converged": True,
                    "periods_run": step_count,
                    "stable_periods": stable_count,
                    "epsilon_final": eps,
                }
                st.session_state[f"{tab_id}_ready_for_training"] = True
        else:
            st.session_state[f"{tab_id}_stable_count"] = 0
            st.session_state[f"{tab_id}_prev_pi1"] = current_pi1
            st.session_state[f"{tab_id}_prev_pi2"] = current_pi2

    # Save checkpoint
    save_checkpoint_econ(config, "step", {"step": step_count})


def run_batch_training_econ(steps_to_run: int, config: dict) -> None:
    """Fast-forward training for N steps (tab-scoped)."""
    tab_id = config.get("tab_id", "default")

    # Check if starting prices have been picked
    starting_prices_picked = st.session_state.get(
        f"{tab_id}_starting_prices_picked", True
    )
    if not starting_prices_picked:
        # Cannot run batch training if starting prices haven't been picked
        return

    # Get configuration
    prices = st.session_state[f"{tab_id}_prices"]
    n_actions = st.session_state[f"{tab_id}_n_actions"]
    t = st.session_state[f"{tab_id}_cfg_t"]
    v = st.session_state[f"{tab_id}_cfg_v"]
    c = st.session_state[f"{tab_id}_cfg_c"]
    alpha = st.session_state[f"{tab_id}_cfg_alpha"]
    delta = st.session_state[f"{tab_id}_cfg_delta"]
    beta = st.session_state[f"{tab_id}_cfg_beta"]
    rng = st.session_state[f"{tab_id}_rng"]
    check_every = st.session_state[f"{tab_id}_cfg_check_every"]
    stable_required = st.session_state[f"{tab_id}_cfg_stable_required"]
    max_periods = st.session_state[f"{tab_id}_cfg_max_periods"]

    # Get current state
    Q1 = st.session_state[f"{tab_id}_Q1"]
    Q2 = st.session_state[f"{tab_id}_Q2"]
    s = st.session_state[f"{tab_id}_current_state"]
    step_count = st.session_state[f"{tab_id}_step_count"]

    # Track starting step for skipped range
    start_step = step_count + 1

    # Use JIT-compiled loop (cap at steps_to_run via max_periods)
    jit_seed = int(rng.integers(0, 2**31))
    prices_arr = np.array(prices, dtype=np.float64)
    jit_max = min(steps_to_run, max_periods - step_count)

    step_ran, s, stable_ct, conv = _train_loop_jit(
        prices_arr, n_actions, c, t, v, alpha, delta, beta,
        Q1, Q2, s, check_every, stable_required, jit_max, jit_seed,
    )
    step_count += step_ran

    if conv:
        st.session_state[f"{tab_id}_convergence_info"] = {
            "converged": True,
            "periods_run": step_count,
            "stable_periods": stable_ct,
            "epsilon_final": epsilon_at(step_count, beta),
        }
    st.session_state[f"{tab_id}_stable_count"] = stable_ct
    st.session_state[f"{tab_id}_prev_pi1"] = np.argmax(Q1, axis=1)
    st.session_state[f"{tab_id}_prev_pi2"] = np.argmax(Q2, axis=1)

    # Update state
    st.session_state[f"{tab_id}_Q1"] = Q1
    st.session_state[f"{tab_id}_Q2"] = Q2
    st.session_state[f"{tab_id}_current_state"] = s
    p1_final, p2_final = index_to_state(s, prices)
    st.session_state[f"{tab_id}_current_p1"] = p1_final
    st.session_state[f"{tab_id}_current_p2"] = p2_final
    st.session_state[f"{tab_id}_step_count"] = step_count

    # Update DataFrame representations
    _fmt = _price_fmt(prices)
    states = [f"s({p1:{_fmt}},{p2:{_fmt}})" for p1 in prices for p2 in prices]
    actions = [f"price={p:{_fmt}}" for p in prices]
    st.session_state[f"{tab_id}_q_table_1"] = pd.DataFrame(
        Q1, index=states, columns=actions
    )
    st.session_state[f"{tab_id}_q_table_2"] = pd.DataFrame(
        Q2, index=states, columns=actions
    )

    # Record skipped steps
    end_step = step_count
    if end_step > start_step:
        st.session_state[f"{tab_id}_skipped_steps"].append((start_step, end_step))

    # Save checkpoint
    save_checkpoint_econ(
        config, "batch", {"steps": steps_to_run, "steps_run": end_step - start_step + 1}
    )


def run_until_convergence_econ(config: dict) -> None:
    """Fast-forward until convergence (stable_required periods) (tab-scoped)."""
    tab_id = config.get("tab_id", "default")

    # Check if starting prices have been picked
    starting_prices_picked = st.session_state.get(
        f"{tab_id}_starting_prices_picked", True
    )
    if not starting_prices_picked:
        # Cannot run until convergence if starting prices haven't been picked
        return

    # Get configuration
    prices = st.session_state[f"{tab_id}_prices"]
    n_actions = st.session_state[f"{tab_id}_n_actions"]
    t = st.session_state[f"{tab_id}_cfg_t"]
    v = st.session_state[f"{tab_id}_cfg_v"]
    c = st.session_state[f"{tab_id}_cfg_c"]
    alpha = st.session_state[f"{tab_id}_cfg_alpha"]
    delta = st.session_state[f"{tab_id}_cfg_delta"]
    beta = st.session_state[f"{tab_id}_cfg_beta"]
    rng = st.session_state[f"{tab_id}_rng"]
    check_every = st.session_state[f"{tab_id}_cfg_check_every"]
    stable_required = st.session_state[f"{tab_id}_cfg_stable_required"]
    max_periods = st.session_state[f"{tab_id}_cfg_max_periods"]

    # Get current state
    Q1 = st.session_state[f"{tab_id}_Q1"]
    Q2 = st.session_state[f"{tab_id}_Q2"]
    s = st.session_state[f"{tab_id}_current_state"]
    step_count = st.session_state[f"{tab_id}_step_count"]

    # Track starting step for skipped range
    start_step = step_count + 1

    # Use a seed derived from the training RNG for the JIT loop
    jit_seed = int(rng.integers(0, 2**31))
    prices_arr = np.array(prices, dtype=np.float64)

    # Run JIT-compiled training loop (600x faster than pure Python)
    step_count, s, stable_count, converged = _train_loop_jit(
        prices_arr, n_actions, c, t, v, alpha, delta, beta,
        Q1, Q2, s, check_every, stable_required, max_periods, jit_seed,
    )

    # Store convergence info
    eps_final = epsilon_at(step_count, beta)
    if converged:
        st.session_state[f"{tab_id}_convergence_info"] = {
            "converged": True,
            "periods_run": step_count,
            "stable_periods": stable_count,
            "epsilon_final": eps_final,
        }
    else:
        st.session_state[f"{tab_id}_convergence_info"] = {
            "converged": False,
            "periods_run": step_count,
            "stable_periods": stable_count,
            "epsilon_final": eps_final,
        }
    st.session_state[f"{tab_id}_stable_count"] = stable_count
    st.session_state[f"{tab_id}_prev_pi1"] = np.argmax(Q1, axis=1)
    st.session_state[f"{tab_id}_prev_pi2"] = np.argmax(Q2, axis=1)

    # Update state
    st.session_state[f"{tab_id}_Q1"] = Q1
    st.session_state[f"{tab_id}_Q2"] = Q2
    st.session_state[f"{tab_id}_current_state"] = s
    p1_final, p2_final = index_to_state(s, prices)
    st.session_state[f"{tab_id}_current_p1"] = p1_final
    st.session_state[f"{tab_id}_current_p2"] = p2_final
    st.session_state[f"{tab_id}_step_count"] = step_count

    # Update DataFrame representations
    _fmt = _price_fmt(prices)
    states = [f"s({p1:{_fmt}},{p2:{_fmt}})" for p1 in prices for p2 in prices]
    actions = [f"price={p:{_fmt}}" for p in prices]
    st.session_state[f"{tab_id}_q_table_1"] = pd.DataFrame(
        Q1, index=states, columns=actions
    )
    st.session_state[f"{tab_id}_q_table_2"] = pd.DataFrame(
        Q2, index=states, columns=actions
    )

    # Record skipped steps
    end_step = step_count
    if end_step > start_step:
        st.session_state[f"{tab_id}_skipped_steps"].append((start_step, end_step))

    # Mark as ready (converged or maxed out)
    st.session_state[f"{tab_id}_ready_for_training"] = True

    # Save checkpoint
    save_checkpoint_econ(
        config,
        "convergence",
        {"convergence_info": st.session_state[f"{tab_id}_convergence_info"]},
    )


def save_checkpoint_econ(config: dict, action_type: str, metadata: dict = None) -> None:
    """Save a checkpoint after user action (single step or batch training)."""
    tab_id = config.get("tab_id", "default")

    checkpoint = {
        "type": action_type,
        "Q1": st.session_state[f"{tab_id}_Q1"].copy(),
        "Q2": st.session_state[f"{tab_id}_Q2"].copy(),
        "q_table_1": st.session_state[f"{tab_id}_q_table_1"].copy(),
        "q_table_2": st.session_state[f"{tab_id}_q_table_2"].copy(),
        "current_state": st.session_state[f"{tab_id}_current_state"],
        "current_p1": st.session_state[f"{tab_id}_current_p1"],
        "current_p2": st.session_state[f"{tab_id}_current_p2"],
        "price_history": st.session_state[f"{tab_id}_price_history"].copy(),
        "skipped_steps": st.session_state[f"{tab_id}_skipped_steps"].copy(),
        "step_count": st.session_state[f"{tab_id}_step_count"],
        "step_log_count": len(st.session_state[f"{tab_id}_step_log"]),
        "convergence_info": st.session_state.get(f"{tab_id}_convergence_info"),
        "ready_for_training": st.session_state.get(
            f"{tab_id}_ready_for_training", True
        ),
        "starting_prices_picked": st.session_state.get(
            f"{tab_id}_starting_prices_picked", True
        ),
        "metadata": metadata or {},
    }

    checkpoints = st.session_state[f"{tab_id}_checkpoints"]
    checkpoints.append(checkpoint)

    # Limit checkpoint history
    if len(checkpoints) > MAX_CHECKPOINTS:
        st.session_state[f"{tab_id}_checkpoints"] = [checkpoints[0]] + checkpoints[
            -(MAX_CHECKPOINTS - 1) :
        ]


def get_display_state_econ(config: dict) -> dict:
    """Get current display state (live or playback) for economics pricing."""
    tab_id = config.get("tab_id", "default")
    playback_idx = st.session_state.get(f"{tab_id}_playback_index", -1)

    if playback_idx < 0:  # Live mode
        price_history = st.session_state[f"{tab_id}_price_history"]
        step_count = st.session_state[f"{tab_id}_step_count"]

        # Filter price_history to only include steps up to step_count
        # This ensures the latest step in price_history matches step_count
        filtered_price_history = [
            item for item in price_history if item[0] <= step_count
        ]

        return {
            "q_table_1": st.session_state[f"{tab_id}_q_table_1"],
            "q_table_2": st.session_state[f"{tab_id}_q_table_2"],
            "current_p1": st.session_state[f"{tab_id}_current_p1"],
            "current_p2": st.session_state[f"{tab_id}_current_p2"],
            "current_state": st.session_state[f"{tab_id}_current_state"],
            "price_history": filtered_price_history,
            "skipped_steps": st.session_state[f"{tab_id}_skipped_steps"],
            "step_log": st.session_state[f"{tab_id}_step_log"],
            "step_count": step_count,
            "convergence_info": st.session_state.get(f"{tab_id}_convergence_info"),
            "ready_for_training": st.session_state.get(
                f"{tab_id}_ready_for_training", True
            ),
            "starting_prices_picked": st.session_state.get(
                f"{tab_id}_starting_prices_picked", True
            ),
            "is_live": True,
        }

    # Historical mode
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]
    if not checkpoints or playback_idx >= len(checkpoints):
        st.session_state[f"{tab_id}_playback_index"] = -1
        return get_display_state_econ(config)

    checkpoint = checkpoints[playback_idx]
    step_log_count = checkpoint.get(
        "step_log_count", len(st.session_state[f"{tab_id}_step_log"])
    )
    filtered_step_log = st.session_state[f"{tab_id}_step_log"][:step_log_count]

    # Filter price_history to only include steps up to step_count for this checkpoint
    # This ensures the latest step in price_history matches step_count
    step_count = checkpoint["step_count"]
    price_history = checkpoint["price_history"]
    filtered_price_history = [item for item in price_history if item[0] <= step_count]

    return {
        "q_table_1": checkpoint["q_table_1"],
        "q_table_2": checkpoint["q_table_2"],
        "current_p1": checkpoint["current_p1"],
        "current_p2": checkpoint["current_p2"],
        "current_state": checkpoint["current_state"],
        "price_history": filtered_price_history,
        "skipped_steps": checkpoint["skipped_steps"],
        "step_log": filtered_step_log,
        "step_count": step_count,
        "convergence_info": checkpoint.get("convergence_info"),
        "ready_for_training": checkpoint.get("ready_for_training", True),
        "starting_prices_picked": checkpoint.get("starting_prices_picked", True),
        "action_type": checkpoint["type"],
        "metadata": checkpoint["metadata"],
        "is_live": False,
    }


def is_in_playback_mode_econ(config: dict) -> bool:
    """Check if currently in playback mode."""
    tab_id = config.get("tab_id", "default")
    return st.session_state.get(f"{tab_id}_playback_index", -1) >= 0


def rewind_checkpoint_econ(config: dict) -> None:
    """Rewind to previous user action checkpoint."""
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]

    if not checkpoints:
        return

    current_idx = st.session_state[f"{tab_id}_playback_index"]

    if current_idx < 0:  # Currently live
        # Store the latest checkpoint index before starting playback
        latest_idx = len(checkpoints) - 1
        st.session_state[f"{tab_id}_latest_checkpoint_index"] = latest_idx
        # Go to the checkpoint before the last one (which is the current live state)
        target_idx = max(0, len(checkpoints) - 2)
        st.session_state[f"{tab_id}_playback_index"] = target_idx
    elif current_idx > 0:
        st.session_state[f"{tab_id}_playback_index"] = current_idx - 1

    # Restore state from checkpoint
    _restore_checkpoint_econ(
        config, checkpoints[st.session_state[f"{tab_id}_playback_index"]]
    )


def forward_checkpoint_econ(config: dict) -> None:
    """Forward to next user action checkpoint."""
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]

    if not checkpoints:
        return

    current_idx = st.session_state[f"{tab_id}_playback_index"]

    if current_idx < 0:  # Currently live, do nothing
        return

    # Get the latest checkpoint index (stored when playback started, or use last checkpoint)
    latest_idx = st.session_state.get(
        f"{tab_id}_latest_checkpoint_index", len(checkpoints) - 1
    )
    latest_idx = min(latest_idx, len(checkpoints) - 1)

    # If we're at or beyond the latest checkpoint, go to live mode
    if current_idx >= latest_idx:
        st.session_state[f"{tab_id}_playback_index"] = -1
        # Live state is already in session state, no need to restore
    else:
        # Move forward one checkpoint
        st.session_state[f"{tab_id}_playback_index"] = current_idx + 1
        _restore_checkpoint_econ(
            config, checkpoints[st.session_state[f"{tab_id}_playback_index"]]
        )


def jump_to_start_econ(config: dict) -> None:
    """Jump to start of playback (step_count = 0, after starting prices picked)."""
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]

    if not checkpoints:
        return

    # Find the checkpoint with step_count = 0 and starting_prices_picked = True
    # This should be the checkpoint after "pick_starting_prices" (Randomised mode)
    # or the init checkpoint (Fixed mode where starting prices are already picked)
    # Default to checkpoint 0 if not found
    target_idx = 0
    for i, checkpoint in enumerate(checkpoints):
        step_count = checkpoint.get("step_count", 0)
        starting_prices_picked = checkpoint.get("starting_prices_picked", True)
        # Look for checkpoint where starting prices have been picked and step_count is 0
        if step_count == 0 and starting_prices_picked:
            target_idx = i
            break

    st.session_state[f"{tab_id}_playback_index"] = target_idx
    _restore_checkpoint_econ(config, checkpoints[target_idx])


def jump_to_latest_econ(config: dict) -> None:
    """Jump back to the latest checkpoint (the last action taken).

    If called from training controls (taking a new action), exits playback mode.
    Otherwise, goes to the latest checkpoint.
    """
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state.get(f"{tab_id}_checkpoints", [])

    if not checkpoints:
        st.session_state[f"{tab_id}_playback_index"] = -1
        return

    # Get the latest checkpoint index (stored when playback started, or use last checkpoint)
    latest_idx = st.session_state.get(
        f"{tab_id}_latest_checkpoint_index", len(checkpoints) - 1
    )
    latest_idx = min(latest_idx, len(checkpoints) - 1)

    # Check if we're currently at the latest checkpoint
    current_idx = st.session_state.get(f"{tab_id}_playback_index", -1)

    # If already at the latest checkpoint or beyond, or in live mode, exit to live mode
    # This handles the case where user is at latest checkpoint and wants to take new actions
    if current_idx >= latest_idx or current_idx < 0:
        st.session_state[f"{tab_id}_playback_index"] = -1
        return

    # Go to the latest checkpoint and restore it
    st.session_state[f"{tab_id}_playback_index"] = latest_idx
    _restore_checkpoint_econ(config, checkpoints[latest_idx])


def _restore_checkpoint_econ(config: dict, checkpoint: dict) -> None:
    """Internal helper to restore state from a checkpoint."""
    tab_id = config.get("tab_id", "default")
    st.session_state[f"{tab_id}_Q1"] = checkpoint["Q1"]
    st.session_state[f"{tab_id}_Q2"] = checkpoint["Q2"]
    st.session_state[f"{tab_id}_q_table_1"] = checkpoint["q_table_1"]
    st.session_state[f"{tab_id}_q_table_2"] = checkpoint["q_table_2"]
    st.session_state[f"{tab_id}_current_state"] = checkpoint["current_state"]
    st.session_state[f"{tab_id}_current_p1"] = checkpoint["current_p1"]
    st.session_state[f"{tab_id}_current_p2"] = checkpoint["current_p2"]
    st.session_state[f"{tab_id}_price_history"] = checkpoint["price_history"]
    st.session_state[f"{tab_id}_skipped_steps"] = checkpoint["skipped_steps"]
    st.session_state[f"{tab_id}_step_count"] = checkpoint["step_count"]
    if "convergence_info" in checkpoint:
        st.session_state[f"{tab_id}_convergence_info"] = checkpoint["convergence_info"]
    st.session_state[f"{tab_id}_ready_for_training"] = checkpoint.get(
        "ready_for_training", True
    )
    st.session_state[f"{tab_id}_starting_prices_picked"] = checkpoint.get(
        "starting_prices_picked", True
    )
