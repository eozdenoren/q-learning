"""Streamlit sidebar & control widgets for dog and bone demo."""

from __future__ import annotations
import streamlit as st
from streamlit_app.state import (
    rewind_checkpoint,
    forward_checkpoint,
    jump_to_latest,
    jump_to_start,
)
from streamlit_app.state_econ import (
    rewind_checkpoint_econ,
    forward_checkpoint_econ,
    jump_to_latest_econ,
    jump_to_start_econ,
    get_display_state_econ,
    calculate_prices,
)

__all__ = [
    "inline_help",
    "parameters_1d",
    "parameters_2d",
    "parameters_econ",
    "playback_controls",
    "playback_controls_econ",
]


def inline_help(text: str, help_text: str) -> None:
    """Display text with an inline help icon"""
    st.markdown(
        f"""
    <style>
    .tooltip-container {{
        position: relative;
        display: inline-block;
    }}
    .tooltip-container .tooltip-text {{
        visibility: hidden;
        width: 220px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        font-weight: normal;
    }}
    .tooltip-container:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    {text}
    <span class="tooltip-container">
        <span style="
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: #6c757d;
            color: white;
            text-align: center;
            line-height: 16px;
            font-size: 11px;
            cursor: help;
            margin-left: 4px;
            vertical-align: middle;
        ">?</span>
        <span class="tooltip-text">{help_text}</span>
    </span>
    """,
        unsafe_allow_html=True,
    )


def q_learning_parameters(tab_id: str) -> dict:
    """Render Q-learning parameters in horizontal layout with two expander rows.

    Returns a dict with keys:
      alpha, gamma, epsilon, reward_val, tab_id
    """

    col_a, col_g, col_e = st.columns([1, 1, 1])

    with col_a:
        alpha: float = st.slider(
            r"$\alpha$ (Learning Rate)", 0.0, 1.0, 0.5, 0.01, key=f"{tab_id}_alpha",
            help="How much Luna adjusts toward new information. High = learn fast, low = learn slowly.",
        )
    with col_g:
        gamma: float = st.slider(
            r"$\gamma$ (Discount Factor)", 0.0, 1.0, 0.9, 0.01, key=f"{tab_id}_gamma",
            help="How much Luna values future rewards. High = patient, low = myopic.",
        )
    with col_e:
        epsilon: float = st.slider(
            r"$\epsilon$ (Exploration Rate)",
            0.0,
            1.0,
            0.2,
            0.01,
            key=f"{tab_id}_epsilon",
            help="How often Luna tries a random action instead of following the Q-table.",
        )

    reward_val = 10.0  # Fixed reward value

    return {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "reward_val": reward_val,
    }


def parameters_1d(tab_id: str) -> dict:
    """Render 1D controls in horizontal layout with two expander rows.

    Returns a dict with keys:
      start_pos, end_pos, goal_pos, start_mode, fixed_start_pos,
      alpha, gamma, epsilon, reward_val, tab_id
    """
    # Row 1: Environment Settings (collapsed by default — most students won't need to change these)
    with st.expander("🐶 Environment Settings", expanded=False):
        # Top row: position inputs side by side
        col1, col2, col3 = st.columns(3)

        with col1:
            start_pos: int = st.number_input(
                "Start Position",
                min_value=-10,
                max_value=10,
                value=0,
                key=f"{tab_id}_start_pos",
            )
        with col2:
            end_pos: int = st.number_input(
                "End Position",
                min_value=-10,
                max_value=10,
                value=5,
                key=f"{tab_id}_end_pos",
            )

        # Ensure end_pos > start_pos
        if end_pos <= start_pos:
            st.error("End Position must be greater than Start Position!")
            end_pos = start_pos + 1

        with col3:
            goal_pos: int = st.number_input(
                "Goal",
                min_value=start_pos,
                max_value=end_pos,
                value=end_pos - 1,
                key=f"{tab_id}_goal_pos",
                help="The position of the bone.",
            )

        # Bottom row: starting mode
        col_mode, col_fixed = st.columns(2)

        with col_mode:
            start_mode: str = st.radio(
                "Starting position for each episode:",
                ["Randomised", "Fixed"],
                index=0,
                key=f"{tab_id}_start_mode",
                horizontal=True,
                help="Choose whether the dog starts at the same fixed position or at a random position for each episode.",
            )

        with col_fixed:
            fixed_start_pos: int = start_pos
            if start_mode == "Fixed":
                fixed_start_pos = st.number_input(
                    "Fixed Start",
                    min_value=start_pos,
                    max_value=end_pos,
                    value=start_pos,
                    key=f"{tab_id}_fixed_start",
                    help="The dog will start at this position for each episode.",
                )

    # Row 2: Q-Learning Parameters
    with st.expander("🧠 Q-Learning Parameters", expanded=True):
        q_learning_params = q_learning_parameters(tab_id)

    return {
        "tab_id": tab_id,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "goal_pos": goal_pos,
        "start_mode": start_mode,
        "fixed_start_pos": fixed_start_pos,
        "alpha": q_learning_params["alpha"],
        "gamma": q_learning_params["gamma"],
        "epsilon": q_learning_params["epsilon"],
        "reward_val": q_learning_params["reward_val"],
    }


def parameters_2d(tab_id: str) -> dict:
    """Render 2D controls in horizontal layout with two expander rows.

    Returns a dict with keys:
      x_start, x_end, y_start, y_end, goal_x, goal_y, start_mode,
      fixed_start_x, fixed_start_y, alpha, gamma, epsilon, reward_val, tab_id
    """
    # Row 1: Environment Settings
    with st.expander("🐶 Environment Settings", expanded=True):
        # Top row: grid range and goal
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            x_start: int = st.number_input(
                "X Start", min_value=-10, max_value=10, value=0, key=f"{tab_id}_x_start"
            )
        with c2:
            x_end: int = st.number_input(
                "X End", min_value=-10, max_value=10, value=3, key=f"{tab_id}_x_end"
            )
        if x_end <= x_start:
            st.error("X End > X Start!")
            x_end = x_start + 1

        with c3:
            y_start: int = st.number_input(
                "Y Start", min_value=-10, max_value=10, value=0, key=f"{tab_id}_y_start"
            )
        with c4:
            y_end: int = st.number_input(
                "Y End", min_value=-10, max_value=10, value=3, key=f"{tab_id}_y_end"
            )
        if y_end <= y_start:
            st.error("Y End > Y Start!")
            y_end = y_start + 1

        with c5:
            goal_x: int = st.number_input(
                "Goal X",
                min_value=x_start,
                max_value=x_end,
                value=x_end - 1,
                key=f"{tab_id}_goal_x",
                help="X coordinate of the bone.",
            )
        with c6:
            goal_y: int = st.number_input(
                "Goal Y",
                min_value=y_start,
                max_value=y_end,
                value=y_end - 1,
                key=f"{tab_id}_goal_y",
                help="Y coordinate of the bone.",
            )

        # Bottom row: starting mode
        col_mode, col_fixed = st.columns(2)

        with col_mode:
            start_mode: str = st.radio(
                "Starting position for each episode:",
                ["Randomised", "Fixed"],
                index=0,
                key=f"{tab_id}_start_mode",
                horizontal=True,
                help="Choose whether the dog starts at the same fixed position or a random position for each episode.",
            )

        with col_fixed:
            fixed_start_x: int = x_start
            fixed_start_y: int = y_start
            if start_mode == "Fixed":
                fc1, fc2 = st.columns(2)
                with fc1:
                    fixed_start_x = st.number_input(
                        "Start X",
                        min_value=x_start,
                        max_value=x_end,
                        value=x_start,
                        key=f"{tab_id}_fixed_start_x",
                    )
                with fc2:
                    fixed_start_y = st.number_input(
                        "Start Y",
                        min_value=y_start,
                        max_value=y_end,
                        value=y_start,
                        key=f"{tab_id}_fixed_start_y",
                    )

    # Row 2: Q-Learning Parameters
    with st.expander("🧠 Q-Learning Parameters", expanded=True):
        q_learning_params = q_learning_parameters(tab_id)

    return {
        "tab_id": tab_id,
        "x_start": x_start,
        "x_end": x_end,
        "y_start": y_start,
        "y_end": y_end,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "start_mode": start_mode,
        "fixed_start_x": fixed_start_x,
        "fixed_start_y": fixed_start_y,
        "alpha": q_learning_params["alpha"],
        "gamma": q_learning_params["gamma"],
        "epsilon": q_learning_params["epsilon"],
        "reward_val": q_learning_params["reward_val"],
    }


def parameters_econ(tab_id: str) -> dict:
    """Render economics pricing controls in horizontal layout with two expander rows.

    Returns a dict with keys:
      tab_id, k1, k2, c, m, alpha, delta, beta, seed, check_every, stable_required, max_periods
    """

    # Default values (used for initial render and Restore Defaults)
    DEFAULTS = {
        f"{tab_id}_alpha": 0.15,
        f"{tab_id}_delta": 0.95,
        f"{tab_id}_t": 1.0,
        f"{tab_id}_m": 15,
        f"{tab_id}_beta_mantissa": 4.0,
        f"{tab_id}_beta_exponent": -6,
        f"{tab_id}_v": 3.0,
        f"{tab_id}_c": 1.0,
    }
    # Seed session state with defaults for any key not yet set
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Main parameters — the four things students experiment with
    col_a, col_d, col_t, col_m = st.columns(4)

    with col_a:
        alpha: float = st.slider(
            r"$\alpha$ (Learning Rate)",
            0.0,
            1.0,
            step=0.001,
            key=f"{tab_id}_alpha",
            help="How fast each firm adjusts to new information.",
        )
    with col_d:
        delta: float = st.slider(
            r"$\gamma$ (Discount Factor)",
            0.0,
            1.0,
            step=0.01,
            key=f"{tab_id}_delta",
            help="How much firms value future profits. High = patient, low = myopic.",
        )
    with col_t:
        _v_cur = st.session_state.get(f"{tab_id}_v", 3.0)
        _c_cur = st.session_state.get(f"{tab_id}_c", 1.0)
        _t_max = round(2.0 * (_v_cur - _c_cur) / 3.0, 1)
        _t_max = max(0.4, _t_max)
        _t_cur = st.session_state.get(f"{tab_id}_t", 1.0)
        if _t_cur > _t_max:
            st.session_state[f"{tab_id}_t"] = _t_max
        t: float = st.slider(
            r"$t$ (Transport Cost)",
            0.3,
            _t_max,
            step=0.1,
            key=f"{tab_id}_t",
            help="Product differentiation. High t = loyal customers, weak competition.",
        )
    with col_m:
        m: int = int(st.radio(
            r"$m$ (Number of Prices)",
            options=[7, 15],
            index=[7, 15].index(st.session_state[f"{tab_id}_m"]),
            key=f"{tab_id}_m",
            horizontal=True,
            help="Number of prices each firm can choose from.",
        ))

    # Benchmarks (always visible)
    prices = None
    start_mode = "Randomised"
    fixed_start_p1 = 2.0
    fixed_start_p2 = 2.0

    # Read v, c from session state (defaults set above)
    v: float = st.session_state.get(f"{tab_id}_v", 3.0)
    c: float = st.session_state.get(f"{tab_id}_c", 1.0)

    if v <= c + 1.5 * t:
        st.warning(
            f"$v$ must be greater than $c + 3t/2 = {c + 1.5*t:.1f}$ for collusion "
            f"to be profitable. Increase $v$ or decrease $t$."
        )

    try:
        prices, p_e, p_c, profit_e, profit_c = calculate_prices(t, v, c, m)
        st.info(
            f"**Nash** $p_e = {p_e:.2f}$, $\\pi_e = {profit_e:.2f}$ | "
            f"**Collusion** $p_c = {p_c:.2f}$, $\\pi_c = {profit_c:.2f}$",
        )
    except Exception as e:
        st.error(f"Error calculating prices: {e}")
        prices = [1.5, 2.0, 2.5, 3.0]
        p_e = 2.0
        p_c = 2.5

    # Advanced parameters — β, v, c, seed, convergence settings
    beta_mantissa: float = st.session_state.get(f"{tab_id}_beta_mantissa", 4.0)
    beta_exponent: int = st.session_state.get(f"{tab_id}_beta_exponent", -6)
    beta: float = beta_mantissa * (10.0 ** beta_exponent)

    with st.expander("Advanced Parameters", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            v = st.number_input(
                r"$v$ (Reservation Value)",
                min_value=0.1, max_value=20.0, step=0.1,
                key=f"{tab_id}_v",
                help="Willingness to pay. Controls collusion ceiling.",
            )
        with adv_col2:
            c = st.number_input(
                r"$c$ (Marginal Cost)",
                min_value=0.0, max_value=10.0, step=0.1,
                key=f"{tab_id}_c",
                help="Marginal cost for both firms.",
            )
        with adv_col3:
            beta_mantissa = st.number_input(
                r"$\beta$ (Exploration Decay)",
                min_value=0.1, max_value=99.9, step=0.1, format="%.1f",
                key=f"{tab_id}_beta_mantissa",
                help="Exploration decays as ε = exp(−β·t). Default 4.0 × 10⁻⁶ works well.",
            )
            beta_exponent = st.number_input(
                r"× 10^",
                min_value=-9, max_value=-1, step=1,
                key=f"{tab_id}_beta_exponent",
            )
            beta = beta_mantissa * (10.0 ** beta_exponent)

        st.markdown("---")
        col_seed, col_check, col_stable, col_max = st.columns(4)
        with col_seed:
            seed: int = st.session_state.get(f"{tab_id}_next_seed", 43)
            st.markdown(f"**Random Seed:** {seed}")
            st.caption("Changes on Reset.")
        with col_check:
            check_every: int = st.number_input(
                "Check Every", min_value=100, max_value=10000, value=1000, step=100,
                key=f"{tab_id}_check_every",
                help="Check policy stability every N steps",
            )
        with col_stable:
            stable_required: int = st.number_input(
                "Stable Required", min_value=1000, max_value=1000000, value=100000, step=10000,
                key=f"{tab_id}_stable_required",
                help="Steps of stable policy required for convergence",
            )
        with col_max:
            max_periods: int = st.number_input(
                "Max Periods", min_value=10000, max_value=10000000, value=5000000, step=100000,
                key=f"{tab_id}_max_periods",
                help="Maximum training steps",
            )

    return {
        "tab_id": tab_id,
        "t": t,
        "v": v,
        "c": c,
        "m": m,
        "alpha": alpha,
        "delta": delta,
        "beta": beta,
        "seed": seed,
        "check_every": check_every,
        "stable_required": stable_required,
        "max_periods": max_periods,
        "start_mode": start_mode,
        "fixed_start_p1": fixed_start_p1,
        "fixed_start_p2": fixed_start_p2,
    }


def playback_controls(config: dict, in_playback: bool) -> None:
    """Render playback controls for navigating through training history.

    Args:
        config: Configuration dict with 'tab_id' key
        in_playback: Whether currently in playback mode
    """
    tab_id = config.get("tab_id", "default")

    col_start, col_prev, col_next, col_latest = st.columns(4)

    checkpoints = st.session_state.get(f"{tab_id}_checkpoints", [])
    playback_idx = st.session_state.get(f"{tab_id}_playback_index", -1)
    # Disable if: no checkpoints, only initial checkpoint (no actions taken), or at first checkpoint
    no_actions_taken = (
        len(checkpoints) <= 1
    )  # Only initial checkpoint exists (no actions yet)
    at_first_checkpoint = playback_idx == 0
    prev_disabled = no_actions_taken or at_first_checkpoint

    with col_start:
        if st.button("⏮", key=f"{tab_id}_start", disabled=prev_disabled, help="Jump to initial state"):
            jump_to_start(config)
            st.rerun()

    with col_prev:
        if st.button("◀", key=f"{tab_id}_rewind", disabled=prev_disabled, help="Previous action"):
            rewind_checkpoint(config)
            st.rerun()

    with col_next:
        if st.button("▶", key=f"{tab_id}_forward", disabled=not in_playback, help="Next action"):
            forward_checkpoint(config)
            st.rerun()

    with col_latest:
        if st.button("⏭", key=f"{tab_id}_latest", disabled=not in_playback, help="Jump to latest"):
            jump_to_latest(config)
            st.rerun()


def playback_controls_econ(config: dict, in_playback: bool) -> None:
    """Render playback controls for navigating through training history (economics version).

    Args:
        config: Configuration dict with 'tab_id' key
        in_playback: Whether currently in playback mode
    """

    tab_id = config.get("tab_id", "default")

    col_start, col_prev, col_next, col_latest = st.columns([1, 1, 1, 1])

    checkpoints = st.session_state.get(f"{tab_id}_checkpoints", [])

    # Get current display state to check step_count
    display_state = get_display_state_econ(config)
    step_count = display_state.get("step_count", 0)

    # Determine button states
    # Prev and Initial: disabled when step_count = 0 or no checkpoints
    no_actions_taken = len(checkpoints) <= 1
    at_step_zero = step_count == 0
    prev_disabled = no_actions_taken or at_step_zero

    # Next and Latest: disabled when not in playback mode OR when at the latest checkpoint
    # Get the latest checkpoint index to check if we're at the latest
    latest_idx = st.session_state.get(
        f"{tab_id}_latest_checkpoint_index", len(checkpoints) - 1 if checkpoints else -1
    )
    latest_idx = min(latest_idx, len(checkpoints) - 1) if checkpoints else -1

    if in_playback:
        playback_idx = st.session_state.get(f"{tab_id}_playback_index", -1)
        # If we're at the latest checkpoint (or beyond), disable Next and Latest
        at_latest_checkpoint = playback_idx >= latest_idx if latest_idx >= 0 else False
        next_disabled = at_latest_checkpoint
        latest_disabled = at_latest_checkpoint
    else:
        # Not in playback mode (live mode) - buttons should be disabled
        next_disabled = True
        latest_disabled = True

    with col_start:
        if st.button("⏮", key=f"{tab_id}_start_econ", disabled=prev_disabled, help="Jump to initial state"):
            jump_to_start_econ(config)
            st.rerun()

    with col_prev:
        if st.button("◀", key=f"{tab_id}_rewind_econ", disabled=prev_disabled, help="Previous action"):
            rewind_checkpoint_econ(config)
            st.rerun()

    with col_next:
        if st.button("▶", key=f"{tab_id}_forward_econ", disabled=next_disabled, help="Next action"):
            forward_checkpoint_econ(config)
            st.rerun()

    with col_latest:
        if st.button("⏭", key=f"{tab_id}_latest_econ", disabled=latest_disabled, help="Jump to latest"):
            jump_to_latest_econ(config)
            st.rerun()
