"""Trajectory computation and display for Economics Pricing Q-learning demo."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_app.state_econ import (
    profit1,
    profit2,
    state_index,
    index_to_state,
    argmax_tie,
)

__all__ = ["render_trajectory_econ", "render_experiment_log", "save_experiment_direct"]

# ---------- Experiment Log ----------

EXPERIMENT_LOG_KEY = "experiment_log"


def _csv_path() -> str:
    """Return path to experiments CSV file."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "experiments.csv")


def _get_log() -> list[dict]:
    """Get experiment log from session state, loading from CSV on first access."""
    if EXPERIMENT_LOG_KEY not in st.session_state:
        path = _csv_path()
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                st.session_state[EXPERIMENT_LOG_KEY] = df.to_dict("records")
            except Exception:
                st.session_state[EXPERIMENT_LOG_KEY] = []
        else:
            st.session_state[EXPERIMENT_LOG_KEY] = []
    return st.session_state[EXPERIMENT_LOG_KEY]


def save_experiment(tab_id: str, delta_1: float, delta_2: float,
                    cycle_length: int, avg_p1: float, avg_p2: float,
                    avg_profit_1: float, avg_profit_2: float,
                    start_p1: float = 0.0, start_p2: float = 0.0,
                    cycle_prices: str = "",
                    transient_prices: str = "") -> None:
    """Save an experiment result to the log."""
    log = _get_log()
    exp = {
        "run": len(log) + 1,
        "alpha": st.session_state.get(f"{tab_id}_cfg_alpha", ""),
        "gamma": st.session_state.get(f"{tab_id}_cfg_delta", ""),
        "beta": f"{st.session_state.get(f'{tab_id}_cfg_beta', 0):.1e}",
        "t": st.session_state.get(f"{tab_id}_cfg_t", ""),
        "v": st.session_state.get(f"{tab_id}_cfg_v", ""),
        "c": st.session_state.get(f"{tab_id}_cfg_c", ""),
        "m": st.session_state.get(f"{tab_id}_cfg_m", ""),
        "seed": st.session_state.get(f"{tab_id}_cfg_seed", ""),
        "p_e": round(st.session_state.get(f"{tab_id}_cfg_p_e", 0), 2),
        "p_c": round(st.session_state.get(f"{tab_id}_cfg_p_c", 0), 2),
        "steps": st.session_state.get(f"{tab_id}_step_count", 0),
        "start_p1": round(start_p1, 2),
        "start_p2": round(start_p2, 2),
        "cycle_len": cycle_length,
        "cycle": cycle_prices,
        "avg_p1": round(avg_p1, 2),
        "avg_p2": round(avg_p2, 2),
        "avg_pi1": round(avg_profit_1, 2),
        "avg_pi2": round(avg_profit_2, 2),
        "Delta1": round(delta_1, 3),
        "Delta2": round(delta_2, 3),
        "path": transient_prices,
    }
    log.append(exp)
    # Auto-save to CSV
    _save_csv(log)


def _save_csv(log: list[dict]) -> None:
    """Save experiment log to CSV file."""
    path = _csv_path()
    if not log:
        if os.path.exists(path):
            os.remove(path)
        return
    pd.DataFrame(log).to_csv(path, index=False)


def save_experiment_direct(
    alpha: float, gamma: float, beta: float, t: float, v: float,
    c: float, m: int, seed: int, steps: int,
    p_e: float, p_c: float,
    start_p1: float, start_p2: float,
    cycle_len: int, cycle_str: str,
    avg_p1: float, avg_p2: float,
    avg_pi1: float, avg_pi2: float,
    delta1: float, delta2: float,
) -> None:
    """Save an experiment result without reading session state."""
    log = _get_log()
    exp = {
        "run": len(log) + 1,
        "alpha": alpha,
        "gamma": gamma,
        "beta": f"{beta:.1e}",
        "t": t,
        "v": v,
        "c": c,
        "m": m,
        "seed": seed,
        "p_e": round(p_e, 2),
        "p_c": round(p_c, 2),
        "steps": steps,
        "start_p1": round(start_p1, 2),
        "start_p2": round(start_p2, 2),
        "cycle_len": cycle_len,
        "cycle": cycle_str,
        "avg_p1": round(avg_p1, 2),
        "avg_p2": round(avg_p2, 2),
        "avg_pi1": round(avg_pi1, 2),
        "avg_pi2": round(avg_pi2, 2),
        "Delta1": round(delta1, 3),
        "Delta2": round(delta2, 3),
        "path": "",
    }
    log.append(exp)
    _save_csv(log)


def delete_experiment(index: int) -> None:
    """Delete an experiment from the log by index."""
    log = _get_log()
    if 0 <= index < len(log):
        log.pop(index)
        # Renumber
        for i, exp in enumerate(log):
            exp["run"] = i + 1
        _save_csv(log)


def render_experiment_log() -> None:
    """Render the experiment log table with delete buttons."""
    log = _get_log()
    if not log:
        return

    st.markdown("---")
    st.subheader("Experiment Log")
    st.caption("Use the Download button below to save your experiment log.")

    df = pd.DataFrame(log)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Action buttons
    col_del, col_clear, col_download, col_spacer = st.columns([1, 1, 1, 2])
    with col_del:
        delete_idx = st.number_input(
            "Delete run #",
            min_value=1,
            max_value=len(log),
            value=len(log),
            step=1,
            key="delete_exp_idx",
        )
        if st.button("Delete", key="delete_exp_btn"):
            delete_experiment(int(delete_idx) - 1)
            st.rerun()
    with col_clear:
        st.markdown("")  # spacing
        st.markdown("")  # spacing
        if st.button("Clear All", key="clear_exp_btn"):
            st.session_state[EXPERIMENT_LOG_KEY] = []
            _save_csv([])
            st.rerun()
    with col_download:
        st.markdown("")  # spacing
        st.markdown("")  # spacing
        csv_data = pd.DataFrame(log).to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="experiments.csv",
            mime="text/csv",
            key="download_exp_btn",
        )


def greedy_successor_econ(
    Q1: np.ndarray,
    Q2: np.ndarray,
    s: int,
    prices: list[float],
    rng: np.random.Generator,
) -> tuple[int, int, float, float, int]:
    """From state index s, take each firm's greedy action and return:
    (a1, a2, p1_next, p2_next, s_next)
    """

    a1 = argmax_tie(Q1[s], rng)
    a2 = argmax_tie(Q2[s], rng)
    p1_next = prices[a1]
    p2_next = prices[a2]
    s_next = state_index(p1_next, p2_next, prices)
    return a1, a2, p1_next, p2_next, s_next


def follow_greedy_until_loop_econ(
    Q1: np.ndarray,
    Q2: np.ndarray,
    start_p1: float,
    start_p2: float,
    prices: list[float],
    rng: np.random.Generator,
    max_steps: int = 100000,
) -> dict:
    """Follow the greedy map from (start_p1, start_p2) until a cycle is detected.

    Returns a dict with the full path and the detected cycle.
    """

    # Start state
    s = state_index(start_p1, start_p2, prices)

    # Remember when we first saw each state (for loop detection)
    first_seen_at = {}  # state_idx -> time step when first visited
    path = []  # list of dicts describing transitions

    for t in range(max_steps):
        if s in first_seen_at:
            loop_start = first_seen_at[s]
            loop_path = path[loop_start:]  # the cycle
            return {
                "path": path,  # all transitions until loop
                "loop_start": loop_start,  # index in path where loop begins
                "loop": loop_path,  # the loop transitions
            }

        # Mark first visit to this state
        first_seen_at[s] = t

        # Decode current state's (p1, p2) for display
        cur_p1, cur_p2 = index_to_state(s, prices)

        # Take greedy actions and move to successor
        a1, a2, p1_next, p2_next, s_next = greedy_successor_econ(Q1, Q2, s, prices, rng)

        # Record transition
        path.append(
            {
                "t": t,
                "state": s,
                "state_str": f"s({cur_p1:.1f},{cur_p2:.1f})",
                "a1_idx": int(a1),
                "a2_idx": int(a2),
                "a1_price": p1_next,  # "max movement read from Q1"
                "a2_price": p2_next,  # "max movement read from Q2"
                "next_state": s_next,
                "next_state_str": f"s({p1_next:.1f},{p2_next:.1f})",
            }
        )

        # Advance
        s = s_next

    # If we hit max_steps without finding a loop
    return {"path": path, "loop_start": None, "loop": []}


def render_trajectory_econ(config: dict) -> None:  # noqa: ARG001
    """Render trajectory computation UI for economics pricing.

    Args:
        config: Configuration dict with 'tab_id' key
        display_state: Display state dict with Q-tables and prices
    """
    tab_id = config.get("tab_id", "default")

    # Get available prices from session state
    prices = st.session_state.get(f"{tab_id}_prices", [])
    if not prices:
        st.info("Initialize the model first to compute trajectories.")
        return
    prices_display = [f"{p:.1f}" for p in prices]

    st.subheader("Pricing Simulation")
    st.markdown(
        rf"""After training, simulate the learned pricing strategies from any starting point.
        Both firms follow their greedy (best-response) policy with no exploration.
        Pick starting prices from $A$ = {{{', '.join(prices_display)}}}.
        """
    )
    # Check if Q-tables have been trained
    step_count = st.session_state.get(f"{tab_id}_step_count", 0)
    if step_count == 0:
        st.info("Train the model first to compute pricing simulations.")
        return

    # Input fields for starting prices
    col1, col2 = st.columns(2)
    with col1:
        start_p1 = st.number_input(
            r"Starting price for Station A ($p_1$):",
            min_value=float(min(prices)),
            max_value=float(max(prices)),
            value=float(prices[len(prices) // 2]),
            step=0.1,
            key=f"{tab_id}_traj_p1",
            help="Starting price for player 1 (Station A)",
        )
    with col2:
        start_p2 = st.number_input(
            r"Starting price for Station B ($p_2$):",
            min_value=float(min(prices)),
            max_value=float(max(prices)),
            value=float(prices[len(prices) // 2]),
            step=0.1,
            key=f"{tab_id}_traj_p2",
            help="Starting price for player 2 (Station B)",
        )

    # Validate and normalize prices to exact values in PRICES list (handles floating-point precision)
    tolerance = 5e-2
    p1_valid = any(abs(start_p1 - p) <= tolerance for p in prices)
    p2_valid = any(abs(start_p2 - p) <= tolerance for p in prices)

    if not p1_valid or not p2_valid:
        st.warning(
            f"⚠️ Prices must be in the action space: {{{', '.join([f'{p:.1f}' for p in prices])}}}. "
            f"Please select valid prices."
        )
        return

    # Normalize to exact values in PRICES to avoid floating-point precision issues
    start_p1 = min(prices, key=lambda p: abs(p - start_p1))
    start_p2 = min(prices, key=lambda p: abs(p - start_p2))

    # Button to compute trajectory
    if st.button("Simulate Pricing", key=f"{tab_id}_compute_traj", type="primary"):
        # Get Q-tables from session state
        Q1 = st.session_state.get(f"{tab_id}_Q1")
        Q2 = st.session_state.get(f"{tab_id}_Q2")
        rng = st.session_state.get(f"{tab_id}_rng")

        if Q1 is None or Q2 is None:
            st.error("Q-tables not initialized. Please initialize the model first.")
            return

        if rng is None:
            rng = np.random.default_rng(43)

        # Compute trajectory
        with st.spinner("Computing trajectory..."):
            traj = follow_greedy_until_loop_econ(
                Q1, Q2, start_p1, start_p2, prices, rng, max_steps=50000
            )

        # Store in session state for display
        st.session_state[f"{tab_id}_trajectory"] = traj
        st.session_state[f"{tab_id}_trajectory_start"] = (start_p1, start_p2)
        st.rerun()

    # Display trajectory if available
    traj = st.session_state.get(f"{tab_id}_trajectory")
    traj_start = st.session_state.get(f"{tab_id}_trajectory_start")

    if traj and traj_start:
        if traj_start != (start_p1, start_p2):
            st.session_state[f"{tab_id}_trajectory"] = None
            st.session_state[f"{tab_id}_trajectory_start"] = None
        else:
            path = traj["path"]
            loop = traj["loop"]
            loop_start = traj["loop_start"]

            # Get parameters
            c = st.session_state.get(f"{tab_id}_cfg_c", 0.0)
            t = st.session_state.get(f"{tab_id}_cfg_t", 1.0)
            v = st.session_state.get(f"{tab_id}_cfg_v", 3.0)
            p_e = st.session_state.get(f"{tab_id}_cfg_p_e", 0.0)
            p_c = st.session_state.get(f"{tab_id}_cfg_p_c", 0.0)
            profit_e = st.session_state.get(f"{tab_id}_cfg_profit_e", 0.0)
            profit_c = st.session_state.get(f"{tab_id}_cfg_profit_c", 0.0)

            if loop_start is not None and loop:
                # Compute cycle averages
                loop_prices_a = [rec["a1_price"] for rec in loop]
                loop_prices_b = [rec["a2_price"] for rec in loop]
                loop_profits_a = [profit1(p1, p2, c, t, v) for p1, p2 in zip(loop_prices_a, loop_prices_b)]
                loop_profits_b = [profit2(p1, p2, c, t, v) for p1, p2 in zip(loop_prices_a, loop_prices_b)]
                avg_p1 = sum(loop_prices_a) / len(loop_prices_a)
                avg_p2 = sum(loop_prices_b) / len(loop_prices_b)
                avg_profit_a = sum(loop_profits_a) / len(loop_profits_a)
                avg_profit_b = sum(loop_profits_b) / len(loop_profits_b)

                denominator = profit_c - profit_e
                if abs(denominator) > 1e-10:
                    delta_a = (avg_profit_a - profit_e) / denominator
                    delta_b = (avg_profit_b - profit_e) / denominator
                else:
                    delta_a = delta_b = float("nan")

                # --- RESULTS: prominent metrics ---
                st.markdown("### Results")

                col_da, col_db = st.columns(2)
                with col_da:
                    if not np.isnan(delta_a):
                        color = "normal" if delta_a > 0.05 else "off"
                        st.metric("Station A: Δ", f"{delta_a:.2f}")
                    st.caption(f"Avg price: {avg_p1:.2f}  |  Avg profit: {avg_profit_a:.3f}")
                with col_db:
                    if not np.isnan(delta_b):
                        st.metric("Station B: Δ", f"{delta_b:.2f}")
                    st.caption(f"Avg price: {avg_p2:.2f}  |  Avg profit: {avg_profit_b:.3f}")

                # Interpretation
                avg_delta = (delta_a + delta_b) / 2 if not (np.isnan(delta_a) or np.isnan(delta_b)) else 0
                if avg_delta > 0.7:
                    st.error(f"**Strong collusion.** Both firms earn well above Nash (Δ ≈ {avg_delta:.2f}).")
                elif avg_delta > 0.3:
                    st.warning(f"**Partial collusion.** Prices are above Nash but below full collusion (Δ ≈ {avg_delta:.2f}).")
                elif avg_delta > 0.05:
                    st.info(f"**Mild supra-competitive pricing.** Slightly above Nash (Δ ≈ {avg_delta:.2f}).")
                else:
                    st.success(f"**Near-competitive.** Close to Nash equilibrium (Δ ≈ {avg_delta:.2f}).")

                # Benchmarks for reference
                st.caption(
                    f"Benchmarks — Nash: p = {p_e:.2f}, π = {profit_e:.3f}  |  "
                    f"Collusion: p = {p_c:.2f}, π = {profit_c:.3f}  |  "
                    f"Cycle length: {len(loop)}"
                )

                # Cycle details in expander
                with st.expander("Show pricing cycle details", expanded=False):
                    cycle_str = " → ".join(
                        f"({rec['a1_price']:.1f}, {rec['a2_price']:.1f})"
                        for rec in loop
                    )
                    st.markdown(f"**Cycle:** {cycle_str}")

                    if path:
                        # Show price trajectory as a simple dataframe
                        full_path = [{"Step": 0, "p₁": start_p1, "p₂": start_p2}]
                        for i, rec in enumerate(path, 1):
                            full_path.append({"Step": i, "p₁": rec["a1_price"], "p₂": rec["a2_price"]})

                        import pandas as pd
                        df = pd.DataFrame(full_path).set_index("Step")
                        st.dataframe(df, height=300)

                # Auto-save to experiment log
                cycle_str = " → ".join(
                    f"({rec['a1_price']:.1f},{rec['a2_price']:.1f})" for rec in loop
                )
                transient_steps = path[:loop_start] if loop_start else []
                transient_str = " → ".join(
                    [f"({start_p1:.1f},{start_p2:.1f})"] +
                    [f"({rec['a1_price']:.1f},{rec['a2_price']:.1f})" for rec in transient_steps]
                )
                save_key = f"{tab_id}_last_saved_sim"
                save_id = (st.session_state.get(f"{tab_id}_step_count", 0), start_p1, start_p2)
                if st.session_state.get(save_key) != save_id:
                    save_experiment(
                        tab_id,
                        delta_1=delta_a, delta_2=delta_b,
                        cycle_length=len(loop),
                        avg_p1=avg_p1, avg_p2=avg_p2,
                        avg_profit_1=avg_profit_a, avg_profit_2=avg_profit_b,
                        start_p1=start_p1, start_p2=start_p2,
                        cycle_prices=cycle_str, transient_prices=transient_str,
                    )
                    st.session_state[save_key] = save_id

            else:
                st.info("No cycle detected. Try training for more steps.")
