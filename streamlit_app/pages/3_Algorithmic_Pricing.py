"""Economics Pricing Example with Q-learning"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root (containing `streamlit_app`) is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import streamlit as st
import pandas as pd
from streamlit_app.ui.controls import parameters_econ
from streamlit_app.ui.trajectory import (
    render_trajectory_econ,
    render_experiment_log,
    follow_greedy_until_loop_econ,
    save_experiment_direct,
)
from streamlit_app.state_econ import (
    init_session_state_econ,
    run_until_convergence_econ,
    get_display_state_econ,
    pick_random_starting_prices_econ,
    run_single_experiment,
    profit1,
    profit2,
    demand1,
    flip_q_table_states,
)

st.set_page_config(page_title="Pricing Strategies in Economics", layout="wide")

st.title("Algorithmic Price Competition with Differentiated Products")
st.markdown(
    """
    Two firms sell substitute products and use Q-learning algorithms to set prices.
    Each firm learns independently, observing only its own profit. What pricing behaviour emerges?
    """
)

tab_1, tab_3, tab_ex = st.tabs(["The Environment", "Simulation", "Exercises"])
# Pricing Battle tab hidden for now — will be introduced later

with tab_1:
    st.markdown(
        r"""
### Two Gas Stations, Each Running Q-Learning

Two gas stations sit on opposite ends of a main road (the Hotelling model from
the lecture). Each station uses its own Q-learning algorithm to set prices.
The algorithms do not communicate — each observes only its own profit.

**State** = last period's price pair $(p_1, p_2)$

**Action** = choose a new price from a grid of $m$ prices

**Reward** = own profit $\pi_i = (p_i - c) \times q_i$

Each firm updates its Q-table independently using the same Bellman rule as Luna:

$$Q_i(s, a_i) \leftarrow Q_i(s, a_i) + \alpha \left[ \pi_i + \gamma \max_{a_i'} Q_i(s', a_i') - Q_i(s, a_i) \right]$$

**The question:** do the algorithms converge to the Nash price ($p_e = c + t$)
or learn to sustain supra-competitive prices closer to the collusive level
($p_c = v - t/2$)?

---

### Measuring the Outcome: The Collusion Index $\Delta$

After training, the algorithms settle into a pricing cycle. We measure where
they ended up using:

$$\Delta_i = \frac{\bar{\pi}_i - \pi_e}{\pi_c - \pi_e}$$

| $\Delta$ | Meaning |
|-----------|---------|
| 0 | Nash (competitive) profit |
| 1 | Full collusion profit |
| Between 0 and 1 | Supra-competitive — partial collusion |
| Negative | Worse than Nash — being exploited |

$\Delta$ is computed separately for each firm and varies across runs due to
randomness in exploration.
        """
    )

    with st.expander("Technical details", expanded=False):
        st.markdown(
            r"""
**Price grid.** Each firm chooses from $m$ equally spaced prices in
$[2p_e - p_c,\; 2p_c - p_e]$, following Calvano et al. (2020).

**Exploration decay.** Exploration rate declines as $\varepsilon_t = e^{-\beta t}$.
Early on, firms explore wildly; over time, they exploit what they've learned.
This is necessary because the environment includes the other firm, which is
also learning — permanent exploration would prevent convergence.
            """
        )

with tab_ex:
    st.markdown(
        r"""
### Instructions

Go to the **Simulation** tab to run experiments. For each exercise:

1. Set the parameters as instructed
2. Click **Train the Model** to run until convergence
3. Click **Simulate Pricing** to see the converged prices and $\Delta$ values
4. The seed changes automatically after each run — click **Reset / Initialize** before the next run

Each simulation is automatically recorded in the **Experiment Log** at the bottom
of the Simulation tab. You can download it as a CSV file at any time.

---

### Exercise 1: Baseline — where do prices end up?

**Parameters:** defaults ($\alpha = 0.15$, $\gamma = 0.95$, $t = 1.0$, $m = 15$).

**Run 3 simulations.** For each, record $\Delta_A$ and $\Delta_B$.

**Report:** What is the average $\Delta$ across your runs? Do the firms converge
to Nash ($\Delta = 0$), full collusion ($\Delta = 1$), or somewhere in between?
How much variation is there across runs?

---

### Exercise 2: The role of patience ($\gamma$)

**Run 3 simulations at each of the following values of $\gamma$:**
- $\gamma = 0.95$ (patient — default)
- $\gamma = 0.50$ (moderately myopic)

Keep all other parameters at their defaults. Reset before each run.

**Report:** How does the average $\Delta$ change when $\gamma$ falls? Does collusion
break down? At what level of $\gamma$ do you first see $\Delta$ close to zero?

> *Hint: if a firm undercuts today, the rival reacts tomorrow. A patient firm
> cares about that retaliation; a myopic firm does not.*

---

### Exercise 3: What sustains collusion?

**Run 1 simulation** with defaults ($\gamma = 0.95$). After training:

- **Simulate** starting from the converged prices. Confirm the cycle is stable.
- **Simulate again** starting from a state where one firm undercuts (set one firm's
  starting price to the Nash price while the other stays at the converged price).

**Report:** Does the rival retaliate? How many steps until prices recover?
The algorithms were never programmed to punish — how does punishment emerge?

---

### Exercise 4: How differentiated are the products?

The parameter $t$ controls switching costs (product differentiation).

**Run 3 simulations at each of:**
- $t = 1.0$ (default — moderate differentiation)
- $t = 0.5$ (low differentiation — easy to switch)

Keep all other parameters at defaults.

**Report:** How does $\Delta$ change with $t$? Is the effect consistent across runs?

---

### Exercise 5: How essential is the product?

The parameter $v$ controls willingness to pay. Open **Advanced Parameters** to change $v$.

**Run 3 simulations at each of:**
- $v = 4.5$ (default)
- $v = 3.0$ (just above the threshold)

**Report:** How does $\Delta$ change? Why might regulators worry more about
algorithmic pricing for essential goods (high $v$)?

---

### Exercise 6: Does the number of prices matter?

The parameter $m$ controls how many prices each firm can choose from.

**Run 3 simulations at each of:**
- $m = 15$ (fine grid — default)
- $m = 7$ (coarse grid)

**Report:** Does a coarser grid make collusion easier or harder? Why?

> *Hint: with fewer prices, the smallest possible undercut is a big price drop.*

---

### Exercise 7: The big picture

No one programmed these firms to collude. They never communicated. Each one
simply maximised its own discounted profit by trial and error.

Based on your experiments:
- **Write down** the key ingredients for supra-competitive pricing to emerge.
- **Answer:** Should competition authorities be concerned about algorithmic pricing?
  What makes this different from human tacit collusion?
        """
    )

with tab_3:

    st.markdown(
        "**1.** Set parameters below. **2.** Click **Train** to run until convergence. "
        "**3.** Scroll down to see the price trajectory and collusion index Δ."
    )

    # Handle Restore Defaults before widgets render
    if st.session_state.get("_restore_defaults", False):
        st.session_state["_restore_defaults"] = False
        # Set all widget keys to their default values
        st.session_state["training_alpha"] = 0.15
        st.session_state["training_delta"] = 0.95
        st.session_state["training_t"] = 1.0
        st.session_state["training_m"] = 15
        st.session_state["training_beta_mantissa"] = 4.0
        st.session_state["training_beta_exponent"] = -6
        st.session_state["training_v"] = 4.5
        st.session_state["training_c"] = 1.0
        st.session_state["training_check_every"] = 1000
        st.session_state["training_stable_required"] = 100000
        st.session_state["training_max_periods"] = 5000000
        st.rerun()

    # --- A. TOP PANEL: Controls ---
    config_demo = parameters_econ("training")

    # Initialize on first load
    if f"{config_demo.get('tab_id', 'training')}_Q1" not in st.session_state:
        config_demo["start_mode"] = "Randomised"
        init_session_state_econ(config_demo)
        pick_random_starting_prices_econ(config_demo)

    # Buttons: Reset, Train, Multi-run, Restore Defaults
    col_reset, col_train, col_multi_n, col_multi_run, col_defaults = st.columns([1.2, 1.2, 0.8, 1.2, 1.2])
    with col_reset:
        if st.button(
            "Reset / Initialize",
            type="secondary",
            key="demo_reset",
            help="Reset the model with the current parameters. Picks random starting prices automatically.",
        ):
            # Random seed so each reset gives a different run
            import os
            new_seed = int.from_bytes(os.urandom(3), "big") % 1000000
            tab_id = config_demo.get("tab_id", "training")
            st.session_state[f"{tab_id}_next_seed"] = new_seed
            config_demo["seed"] = new_seed
            config_demo["start_mode"] = "Randomised"
            init_session_state_econ(config_demo)
            pick_random_starting_prices_econ(config_demo)
            st.rerun()
    with col_train:
        if st.button(
            "Train the Model",
            type="primary",
            key="demo_train",
            help="Run training until the greedy policy is stable, then stop.",
        ):
            # Ensure starting prices are picked
            tab_id = config_demo.get("tab_id", "training")
            if not st.session_state.get(f"{tab_id}_starting_prices_picked", False):
                pick_random_starting_prices_econ(config_demo)
            run_until_convergence_econ(config_demo)
            # Auto-change seed so next Reset gives a different run
            import os
            new_seed = int.from_bytes(os.urandom(3), "big") % 1000000
            st.session_state[f"{tab_id}_next_seed"] = new_seed
            st.rerun()
    with col_multi_n:
        n_runs = st.number_input(
            "# runs",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="multi_n_runs",
        )
    with col_multi_run:
        run_multi = st.button(
            f"Train {n_runs} Seeds",
            type="primary",
            key="multi_run_btn",
            help="Train the model with multiple random seeds using the current parameters. Each run is saved to the experiment log.",
        )
    with col_defaults:
        if st.button(
            "Restore Defaults",
            type="secondary",
            key="demo_defaults",
            help="Reset all parameters to their default values.",
        ):
            st.session_state["_restore_defaults"] = True
            st.rerun()

    if run_multi:
        import os
        tab_id = config_demo.get("tab_id", "training")
        cfg_alpha = st.session_state.get(f"{tab_id}_cfg_alpha", 0.15)
        cfg_gamma = st.session_state.get(f"{tab_id}_cfg_delta", 0.95)
        cfg_beta = st.session_state.get(f"{tab_id}_cfg_beta", 4e-6)
        cfg_t = st.session_state.get(f"{tab_id}_cfg_t", 1.0)
        cfg_v = st.session_state.get(f"{tab_id}_cfg_v", 3.0)
        cfg_c = st.session_state.get(f"{tab_id}_cfg_c", 1.0)
        cfg_m = st.session_state.get(f"{tab_id}_cfg_m", 15)

        progress = st.progress(0, text="Starting...")
        for i in range(n_runs):
            seed = int.from_bytes(os.urandom(3), "big") % 1000000
            progress.progress(
                (i) / n_runs,
                text=f"Run {i + 1}/{n_runs} (seed {seed})..."
            )
            result = run_single_experiment(
                c=cfg_c, t=cfg_t, v=cfg_v, m=cfg_m,
                alpha=cfg_alpha, gamma=cfg_gamma, beta=cfg_beta,
                seed=seed,
            )
            save_experiment_direct(
                alpha=cfg_alpha, gamma=cfg_gamma, beta=cfg_beta,
                t=cfg_t, v=cfg_v, c=cfg_c, m=cfg_m, seed=seed,
                steps=result["steps"],
                p_e=result["p_e"], p_c=result["p_c"],
                start_p1=result["start_p1"], start_p2=result["start_p2"],
                cycle_len=result["cycle_len"], cycle_str=result["cycle_str"],
                avg_p1=result["avg_p1"], avg_p2=result["avg_p2"],
                avg_pi1=result["avg_pi1"], avg_pi2=result["avg_pi2"],
                delta1=result["delta1"], delta2=result["delta2"],
            )
        progress.progress(1.0, text="Done!")
        st.rerun()

    # Show convergence info
    display_state = get_display_state_econ(config_demo)
    convergence_info = display_state.get("convergence_info")
    step_count = display_state.get("step_count", 0)

    if convergence_info:
        if convergence_info.get("converged", False):
            st.success(
                f"Converged after {convergence_info['periods_run']:,} steps. "
                f"Policy stable for {convergence_info['stable_periods']:,} steps. "
                f"Final exploration rate: {convergence_info['epsilon_final']:.2e}"
            )
        else:
            st.warning(
                f"Not converged after {convergence_info['periods_run']:,} steps. "
                f"Stable for {convergence_info['stable_periods']:,} steps. "
                f"Final exploration rate: {convergence_info['epsilon_final']:.2e}"
            )
    elif step_count == 0:
        st.info("Click **Train the Model** to start training.")

    st.markdown("---")

    # --- B. Q-MATRICES ---
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        with st.expander(r"Q-Matrix: Station A ($Q_1$)", expanded=False):
            st.dataframe(
                display_state["q_table_1"].style.highlight_max(
                    axis=1, color="lightgreen"
                ),
                width="stretch",
                height=400,
            )

    with col_q2:
        with st.expander(r"Q-Matrix: Station B ($Q_2$)", expanded=False):
            st.dataframe(
                display_state["q_table_2"].style.highlight_max(
                    axis=1, color="lightgreen"
                ),
                width="stretch",
                height=400,
            )

    st.markdown("---")

    # --- C. PRICING SIMULATION ---
    render_trajectory_econ(config_demo)

    # --- D. EXPERIMENT LOG ---
    render_experiment_log()



# --- Pricing Battle tab hidden for now ---
# To re-enable, uncomment below and add tab_4 back to st.tabs()
# with tab_4:
#     st.header("Pricing Battle")
#     st.markdown(
#         """
#         Export your Q-tables to a csv file (from the Demo page) and upload it to this pricing battle page, and compete with other players. The winner is the one with the highest average profit of the prices/profits cycle.
#         
#         **Important:** When uploading a Q-table, you must specify whether it was trained from Player 1's perspective (Q1) or Player 2's perspective (Q2). 
#         The state encoding s(p1, p2) depends on which player's perspective the Q-table was trained from. If you upload a Q-table with the wrong perspective, 
#         the system will automatically flip the state indices to match the battle role.
#         """
#     )
#     # Initialize perspective variables (will be set when files are uploaded)
#     q1_perspective = None
#     q2_perspective = None
# 
#     col_q1, col_q2 = st.columns(2)
#     with col_q1:
#         # upload 1st Q-table
#         player_1_name = st.text_input(
#             "Enter your name", value="Station A", key="player_1_name"
#         )
#         uploaded_file1 = st.file_uploader(
#             "Upload your 1st Q-tables csv file", type="csv", key="q_table_upload_1"
#         )
#         if uploaded_file1 is not None:
#             # Reset file pointer and read
#             uploaded_file1.seek(0)
#             df1_display = pd.read_csv(uploaded_file1)
#             st.dataframe(df1_display)
#             q1_perspective = st.selectbox(
#                 "This Q-table is from the perspective of:",
#                 ["Player 1 (Q1)", "Player 2 (Q2)"],
#                 key="q1_perspective",
#                 help="Select whether this Q-table was trained from Player 1's perspective (Q1) or Player 2's perspective (Q2). This will be used to correctly interpret the state encoding.",
#             )
# 
#     with col_q2:
#         # upload 2nd Q-table
#         player_2_name = st.text_input(
#             "Enter your name", value="Station B", key="player_2_name"
#         )
#         uploaded_file2 = st.file_uploader(
#             "Upload your 2nd Q-tables csv file", type="csv", key="q_table_upload_2"
#         )
#         if uploaded_file2 is not None:
#             # Reset file pointer and read
#             uploaded_file2.seek(0)
#             df2_display = pd.read_csv(uploaded_file2)
#             st.dataframe(df2_display)
#             q2_perspective = st.selectbox(
#                 "This Q-table is from the perspective of:",
#                 ["Player 1 (Q1)", "Player 2 (Q2)"],
#                 key="q2_perspective",
#                 help="Select whether this Q-table was trained from Player 1's perspective (Q1) or Player 2's perspective (Q2). This will be used to correctly interpret the state encoding.",
#             )
# 
#     st.markdown("---")
# 
#     # Environment parameters for the battle (must match the Q-tables)
#     st.subheader("Environment Parameters")
#     st.info(
#         "⚠️ These parameters must match the ones used to train the uploaded Q-tables."
#     )
# 
#     col_t, col_v, col_c = st.columns(3)
#     with col_t:
#         battle_t = st.number_input(
#             r"$t$ (Transport Cost)",
#             min_value=0.1,
#             max_value=10.0,
#             value=1.0,
#             step=0.1,
#             key="battle_t",
#             help="Transport cost / degree of product differentiation",
#         )
#     with col_v:
#         battle_v = st.number_input(
#             r"$v$ (Reservation Value)",
#             min_value=0.1,
#             max_value=20.0,
#             value=3.0,
#             step=0.1,
#             key="battle_v",
#             help="Maximum willingness to pay",
#         )
#     with col_c:
#         battle_c = st.number_input(
#             r"$c$ (Marginal Cost)",
#             min_value=0.0,
#             max_value=10.0,
#             value=1.0,
#             step=0.1,
#             key="battle_c",
#             help="Marginal cost for both players",
#         )
# 
#     st.markdown("---")
# 
#     # Starting prices for the battle
#     st.subheader("Starting Prices for Battle")
#     col_start_p1, col_start_p2 = st.columns(2)
#     with col_start_p1:
#         battle_start_p1 = st.number_input(
#             "Starting price for Player 1",
#             min_value=0.1,
#             max_value=50.0,
#             value=2.0,
#             step=0.1,
#             key="battle_start_p1",
#         )
#     with col_start_p2:
#         battle_start_p2 = st.number_input(
#             "Starting price for Player 2",
#             min_value=0.1,
#             max_value=50.0,
#             value=2.0,
#             step=0.1,
#             key="battle_start_p2",
#         )
# 
#     st.markdown("---")
# 
#     if st.button(
#         "Compute Trajectory & Determine Winner",
#         key="compute_trajectory",
#         type="primary",
#     ):
#         # Check if both files are uploaded
#         if uploaded_file1 is None or uploaded_file2 is None:
#             st.error(
#                 "❌ Please upload both Q-table CSV files before computing the trajectory."
#             )
#             st.stop()
# 
#         try:
#             # Reset file pointers and load Q-tables from CSV
#             uploaded_file1.seek(0)
#             uploaded_file2.seek(0)
#             df1 = pd.read_csv(uploaded_file1, index_col=0)
#             df2 = pd.read_csv(uploaded_file2, index_col=0)
# 
#             # Extract prices from column names (format: "price=X.X")
#             def extract_prices_from_columns(df):
#                 prices = []
#                 for col in df.columns:
#                     if "price=" in col:
#                         price_str = col.replace("price=", "")
#                         try:
#                             prices.append(float(price_str))
#                         except ValueError:
#                             pass
#                 return sorted(prices)
# 
#             prices1 = extract_prices_from_columns(df1)
#             prices2 = extract_prices_from_columns(df2)
# 
#             # Validate that both Q-tables use the same prices
#             if len(prices1) == 0 or len(prices2) == 0:
#                 st.error(
#                     "❌ Could not extract prices from Q-table columns. Expected format: 'price=X.X'"
#                 )
#                 st.stop()
# 
#             if prices1 != prices2:
#                 st.warning(
#                     "⚠️ The two Q-tables use different price sets. Using prices from the first Q-table."
#                 )
#                 st.info(f"Q-table 1 prices: {[f'{p:.1f}' for p in prices1]}")
#                 st.info(f"Q-table 2 prices: {[f'{p:.1f}' for p in prices2]}")
#                 prices = prices1
#             else:
#                 prices = prices1
# 
#             # Validate starting prices are in prices and normalize to exact values, to avoid floating-point precision issues
#             tolerance = 1e-3
#             p1_valid = any(abs(battle_start_p1 - p) < tolerance for p in prices)
#             p2_valid = any(abs(battle_start_p2 - p) < tolerance for p in prices)
# 
#             if not p1_valid or not p2_valid:
#                 st.error(
#                     f"❌ Starting prices must be within tolerance of prices in the action space. "
#                     f"Valid prices: {[f'{p:.1f}' for p in prices]}"
#                 )
#                 st.stop()
# 
#             # Store original values for comparison
#             original_p1 = battle_start_p1
#             original_p2 = battle_start_p2
# 
#             # Always normalize to exact values from 'prices' to avoid floating-point precision issues
#             battle_start_p1 = min(prices, key=lambda p: abs(p - original_p1))
#             battle_start_p2 = min(prices, key=lambda p: abs(p - original_p2))
# 
#             # Only show info if values were actually adjusted
#             if (
#                 abs(battle_start_p1 - original_p1) > 1e-10
#                 or abs(battle_start_p2 - original_p2) > 1e-10
#             ):
#                 st.info(
#                     f"ℹ️ Starting prices normalized to: p1={battle_start_p1:.1f}, p2={battle_start_p2:.1f}"
#                 )
# 
#             # Validate Q-table dimensions
#             n_actions = len(prices)
#             n_states = n_actions * n_actions
# 
#             # Calculate equilibrium and collusion prices and profits (Hotelling)
#             from streamlit_app.state_econ import calculate_prices as _calc_prices
#             _, p_e, p_c, profit_e, profit_c = _calc_prices(battle_t, battle_v, battle_c, 15)
# 
#             if df1.shape != (n_states, n_actions) or df2.shape != (n_states, n_actions):
#                 st.error(
#                     f"❌ Q-table dimensions don't match expected size. "
#                     f"Expected: ({n_states}, {n_actions}), "
#                     f"Got: Q-table 1: {df1.shape}, Q-table 2: {df2.shape}"
#                 )
#                 st.stop()
# 
#             # Check if perspective selection was made for both Q-tables
#             if q1_perspective is None or q2_perspective is None:
#                 st.error(
#                     "❌ Please select the perspective for both uploaded Q-tables before computing the trajectory."
#                 )
#                 st.stop()
# 
#             # Convert DataFrames to numpy arrays
#             Q1 = df1.values.copy()
#             Q2 = df2.values.copy()
# 
#             # Apply state flipping based on perspective vs battle role
#             # Q1 is used for Player 1: flip if uploaded Q-table is from Player 2's perspective
#             if q1_perspective == "Player 2 (Q2)":
#                 Q1 = flip_q_table_states(Q1, prices)
#                 st.info(
#                     "ℹ️ Q-table 1 was flipped because it was from Player 2's perspective but is being used for Player 1."
#                 )
# 
#             # Q2 is used for Player 2: flip if uploaded Q-table is from Player 1's perspective
#             if q2_perspective == "Player 1 (Q1)":
#                 Q2 = flip_q_table_states(Q2, prices)
#                 st.info(
#                     "ℹ️ Q-table 2 was flipped because it was from Player 1's perspective but is being used for Player 2."
#                 )
# 
#             # Initialize random number generator
#             rng = np.random.default_rng(43)
# 
#             # Compute trajectory
#             with st.spinner("🔄 Computing trajectory and detecting cycle..."):
#                 traj = follow_greedy_until_loop_econ(
#                     Q1,
#                     Q2,
#                     battle_start_p1,
#                     battle_start_p2,
#                     prices,
#                     rng,
#                     max_steps=50000,
#                 )
# 
#             # Store in session state for potential display (use different keys to avoid widget conflicts)
#             st.session_state["battle_trajectory"] = traj
#             st.session_state["battle_prices"] = prices
#             st.session_state["battle_cfg_t"] = battle_t
#             st.session_state["battle_cfg_v"] = battle_v
#             st.session_state["battle_cfg_c"] = battle_c
#             st.session_state["battle_cfg_start"] = (battle_start_p1, battle_start_p2)
#             st.session_state["battle_cfg_p_e"] = p_e
#             st.session_state["battle_cfg_p_c"] = p_c
#             st.session_state["battle_cfg_profit_e"] = profit_e
#             st.session_state["battle_cfg_profit_c"] = profit_c
# 
#             st.rerun()
# 
#         except Exception as e:
#             st.error(f"❌ Error computing trajectory: {e}")
#             st.exception(e)
# 
#     # Display results if trajectory has been computed
#     if "battle_trajectory" in st.session_state:
#         traj = st.session_state["battle_trajectory"]
#         prices = st.session_state["battle_prices"]
#         # Read stored config values (stored when trajectory was computed)
#         battle_t = st.session_state.get("battle_cfg_t", 1.0)
#         battle_v = st.session_state.get("battle_cfg_v", 3.0)
#         battle_c = st.session_state.get("battle_cfg_c", 1.0)
#         battle_start = st.session_state.get("battle_cfg_start", (7.0, 7.0))
#         battle_start_p1, battle_start_p2 = battle_start
#         p_e = st.session_state.get("battle_cfg_p_e", 0.0)
#         p_c = st.session_state.get("battle_cfg_p_c", 0.0)
#         profit_e = st.session_state.get("battle_cfg_profit_e", 0.0)
#         profit_c = st.session_state.get("battle_cfg_profit_c", 0.0)
# 
#         path = traj["path"]
#         loop = traj["loop"]
#         loop_start = traj["loop_start"]
# 
#         st.markdown("---")
#         st.markdown(
#             rf"""
#             - Equilibrium price: $p_e = {p_e:.2f}$
#             - Collusion price: $p_c = {p_c:.2f}$
#             - Equilibrium profit: $\pi_e = {profit_e:.2f}$
#             - Collusion profit: $\pi_c = {profit_c:.2f}$
#             """
#         )
# 
#         st.subheader("🏆 Battle Results")
# 
#         if loop_start is not None and loop:
#             # Calculate average profits in the cycle
#             loop_profits_alice = []
#             loop_profits_bob = []
#             loop_prices_alice = []
#             loop_prices_bob = []
# 
#             for rec in loop:
#                 p1 = rec["a1_price"]
#                 p2 = rec["a2_price"]
#                 loop_prices_alice.append(p1)
#                 loop_prices_bob.append(p2)
#                 pi1 = profit1(p1, p2, battle_c, battle_t, battle_v)
#                 pi2 = profit2(p1, p2, battle_c, battle_t, battle_v)
#                 loop_profits_alice.append(pi1)
#                 loop_profits_bob.append(pi2)
# 
#             avg_profit_alice = (
#                 sum(loop_profits_alice) / len(loop_profits_alice)
#                 if loop_profits_alice
#                 else 0.0
#             )
#             avg_profit_bob = (
#                 sum(loop_profits_bob) / len(loop_profits_bob)
#                 if loop_profits_bob
#                 else 0.0
#             )
#             avg_p1 = (
#                 sum(loop_prices_alice) / len(loop_prices_alice)
#                 if loop_prices_alice
#                 else 0.0
#             )
#             avg_p2 = (
#                 sum(loop_prices_bob) / len(loop_prices_bob) if loop_prices_bob else 0.0
#             )
# 
#             # Determine winner
#             if avg_profit_alice > avg_profit_bob:
#                 winner = f"👩🏼‍💼 {player_1_name} (Player 1)"
#                 winner_profit = avg_profit_alice
#                 loser_profit = avg_profit_bob
#             elif avg_profit_bob > avg_profit_alice:
#                 winner = f"🧑🏼‍💼 {player_2_name} (Player 2)"
#                 winner_profit = avg_profit_bob
#                 loser_profit = avg_profit_alice
#             else:
#                 winner = "🤝 Tie"
#                 winner_profit = avg_profit_alice
#                 loser_profit = avg_profit_bob
# 
#             # Display winner with style
#             if winner != "🤝 Tie":
#                 st.success(f"## 🏆 Winner: {winner}!")
#                 st.markdown(
#                     f"""
#                     **Average Profit: {winner_profit:.2f}** (vs {loser_profit:.2f})
#                     """
#                 )
#             else:
#                 st.info(f"## {winner}!")
#                 st.markdown(rf"**Average Profit: {winner_profit:.2f}** (both players)")
# 
#             # Calculate normalized profits
#             denominator = profit_c - profit_e
#             if abs(denominator) > 1e-10:  # Avoid division by zero
#                 normalized_profit_alice = (avg_profit_alice - profit_e) / denominator
#                 normalized_profit_bob = (avg_profit_bob - profit_e) / denominator
#             else:
#                 normalized_profit_alice = None
#                 normalized_profit_bob = None
# 
#             # Display detailed results
#             col_result1, col_result2 = st.columns(2)
# 
#             with col_result1:
#                 st.metric(
#                     f"👩🏼‍💼 {player_1_name} (Player 1)",
#                     f"{avg_profit_alice:.2f}",
#                     delta=(
#                         f"{avg_profit_alice - avg_profit_bob:.2f}"
#                         if avg_profit_alice != avg_profit_bob
#                         else None
#                     ),
#                     help=rf"Average price: $\bar{{p}}_1 = {avg_p1:.2f}$, average profit: $\bar{{\pi}}_1 = {avg_profit_alice:.2f}$",
#                 )
#                 if normalized_profit_alice is not None:
#                     st.metric(
#                         "Normalised average profit",
#                         f"{normalized_profit_alice:.2f}",
#                         help=r"Normalised average profit: $\Delta_1 = \dfrac{{\bar{{\pi}}_1 - \pi_e}}{{\pi_c - \pi_e}}$",
#                     )
#                 else:
#                     st.metric(
#                         "Normalised average profit",
#                         "N/A",
#                         help="Cannot calculate normalized profit: denominator (collusion profit - equilibrium profit) is zero",
#                     )
# 
#             with col_result2:
#                 st.metric(
#                     f"🧑🏼‍💼 {player_2_name} (Player 2)",
#                     f"{avg_profit_bob:.2f}",
#                     delta=(
#                         f"{avg_profit_bob - avg_profit_alice:.2f}"
#                         if avg_profit_bob != avg_profit_alice
#                         else None
#                     ),
#                     help=rf"Average price: $\bar{{p}}_2 = {avg_p2:.2f}$, average profit: $\bar{{\pi}}_2 = {avg_profit_bob:.2f}$",
#                 )
#                 if normalized_profit_bob is not None:
#                     st.metric(
#                         "Normalised average profit",
#                         f"{normalized_profit_bob:.2f}",
#                         help=r"Normalised average profit: $\Delta_2 = \dfrac{{\bar{{\pi}}_2 - \pi_e}}{{\pi_c - \pi_e}}$",
#                     )
#                 else:
#                     st.metric(
#                         "Normalised average profit",
#                         "N/A",
#                         help="Cannot calculate normalized profit: denominator (collusion profit - equilibrium profit) is zero",
#                     )
# 
#             # Display cycle information
#             st.markdown("---")
#             st.markdown("### 📊 Cycle Information")
# 
#             # Display price trajectory (same format as tab_2)
#             st.markdown(r"**Trajectory Steps for $p_1, p_2$ :**")
# 
#             if path:
#                 # Add step 0 with starting prices at the beginning
#                 full_path = [
#                     {
#                         "step_num": 0,
#                         "a1_price": battle_start_p1,
#                         "a2_price": battle_start_p2,
#                     }
#                 ]
#                 # Add path steps, renumbering them starting from 1
#                 for i, rec in enumerate(path, start=1):
#                     full_path.append(
#                         {
#                             "step_num": i,
#                             "a1_price": rec["a1_price"],
#                             "a2_price": rec["a2_price"],
#                         }
#                     )
# 
#                 # Build table data: rows are Station A, Station B, Step; columns are trajectory steps
#                 alice_row = []
#                 bob_row = []
#                 step_row = []
# 
#                 # Fixed column width: calculate width to show 10 columns consistently
#                 num_visible_cols = 10
#                 fixed_col_width = "90px"  # Fixed width per column
# 
#                 for rec in full_path:
#                     alice_row.append(f"{rec['a1_price']:.1f}")
#                     bob_row.append(f"{rec['a2_price']:.1f}")
#                     step_row.append(f"Step {rec['step_num']}")
# 
#                 # Create HTML table with horizontal scroll, no header, fixed column width
#                 table_id = f"battle_trajectory_table_{id(path)}"
#                 html_table = f"""
#                 <div id="{table_id}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
#                     <table style="border-collapse: collapse; table-layout: fixed;">
#                         <tbody>
#                             <tr>
#                                 <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 {player_1_name}</td>
#                                 {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_row])}
#                             </tr>
#                             <tr>
#                                 <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 {player_2_name}</td>
#                                 {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in bob_row])}
#                             </tr>
#                             <tr>
#                                 <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">Step</td>
#                                 {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 0.85em; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in step_row])}
#                             </tr>
#                         </tbody>
#                     </table>
#                 </div>
#                 """
# 
#                 st.markdown(html_table, unsafe_allow_html=True)
# 
#                 if len(full_path) > num_visible_cols:
#                     st.caption(
#                         "💡 Scroll horizontally to see all steps. The newest steps appear on the left."
#                     )
# 
#             # Trajectory table for profit steps (same format as tab_2)
#             st.markdown(r"**Trajectory Steps for $\pi_1, \pi_2$ :**")
# 
#             if path:
#                 # Calculate profits for each step in full_path
#                 alice_profit_row = []
#                 bob_profit_row = []
#                 step_row_profit = []
# 
#                 for rec in full_path:
#                     p1 = rec["a1_price"]
#                     p2 = rec["a2_price"]
#                     pi1 = profit1(p1, p2, battle_c, battle_t, battle_v)
#                     pi2 = profit2(p1, p2, battle_c, battle_t, battle_v)
#                     alice_profit_row.append(f"{pi1:.2f}")
#                     bob_profit_row.append(f"{pi2:.2f}")
#                     step_row_profit.append(f"Step {rec['step_num']}")
# 
#                 # Create HTML table for profits with horizontal scroll, fixed column width
#                 table_id_profit = f"battle_trajectory_profit_table_{id(path)}"
#                 html_table_profit = f"""
#                 <div id="{table_id_profit}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
#                     <table style="border-collapse: collapse; table-layout: fixed;">
#                         <tbody>
#                             <tr>
#                                 <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 {player_1_name}</td>
#                                 {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_profit_row])}
#                             </tr>
#                             <tr>
#                                 <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 {player_2_name}</td>
#                                 {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in bob_profit_row])}
#                             </tr>
#                             <tr>
#                                 <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">Step</td>
#                                 {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 0.85em; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in step_row_profit])}
#                             </tr>
#                         </tbody>
#                     </table>
#                 </div>
#                 """
# 
#                 st.markdown(html_table_profit, unsafe_allow_html=True)
# 
#                 if len(full_path) > num_visible_cols:
#                     st.caption(
#                         "💡 Scroll horizontally to see all steps. The newest steps appear on the right."
#                     )
#             if (
#                 normalized_profit_alice is not None
#                 and normalized_profit_bob is not None
#             ):
#                 st.markdown(
#                     rf"""
#                     - **Cycle detected**: starts at step {loop_start}, length = {len(loop)} steps
#                     - **Average price for {player_1_name} ($\bar{{p}}_1$)**: {avg_p1:.2f}
#                     - **Average price for {player_2_name} ($\bar{{p}}_2$)**: {avg_p2:.2f}
#                     - **Average profit for {player_1_name} ($\bar{{\pi}}_1$)**: {avg_profit_alice:.2f}
#                     - **Average profit for {player_2_name} ($\bar{{\pi}}_2$)**: {avg_profit_bob:.2f}
#                     - Equilibrium profit: $\pi_e = {profit_e:.2f}$
#                     - Collusion profit: $\pi_c = {profit_c:.2f}$
#                     - **Normalised average profit for {player_1_name} ($\Delta_1$)**= $\dfrac{{{avg_profit_alice:.2f}-{profit_e:.2f}}}{{{profit_c:.2f} - {profit_e:.2f}}}$ = {normalized_profit_alice:.2f}
#                     - **Normalised average profit for {player_2_name} ($\Delta_2$)**= $\dfrac{{{avg_profit_bob:.2f}-{profit_e:.2f}}}{{{profit_c:.2f} - {profit_e:.2f}}}$ = {normalized_profit_bob:.2f}
#                     """
#                 )
#             else:
#                 st.markdown(
#                     rf"""
#                     - **Cycle detected**: starts at step {loop_start}, length = {len(loop)} steps
#                     - **Average price for {player_1_name} ($p_1$)**: {avg_p1:.2f}
#                     - **Average price for {player_2_name} ($p_2$)**: {avg_p2:.2f}
#                     - **Average profit for {player_1_name} ($\pi_1$)**: {avg_profit_alice:.2f}
#                     - **Average profit for {player_2_name} ($\pi_2$)**: {avg_profit_bob:.2f}
#                     - **Normalised profit**: Cannot be calculated (denominator is zero)
#                     """
#                 )
#         else:
#             st.warning(
#                 "⚠️ No cycle detected within max_steps. Consider adjusting starting prices or increasing max_steps."
#             )
# 
# st.markdown("---")
# st.markdown(
#     '<p style="text-align: center; color: grey; font-size: 0.85em;">'
#     "&copy; 2026 Emre Ozdenoren. All rights reserved."
#     "</p>",
#     unsafe_allow_html=True,
# )
