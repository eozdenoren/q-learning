"""Dog & Bone Q-learning demo page — simplified 1D version."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (containing `streamlit_app`) is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
from streamlit_app.ui.controls import parameters_1d
from streamlit_app.ui.training import render_training_controls
from streamlit_app.ui.grid import render_grid_1d
from streamlit_app.ui.charts import render_q_history_chart, render_steps_chart
from streamlit_app.ui.policy_plot import render_policy_1d
from streamlit_app.state import (
    get_display_state,
    init_session_state,
    reset_episode,
    step_agent,
    run_batch_training,
)

st.set_page_config(page_title="The Dog & The Bone – Q-Learning", layout="wide")

st.title("The Dog & The Bone")

st.markdown(
    """
    Use Q-learning to help Luna the robot dog find the bone.
    Step through the training process to watch the Q-table fill in, or fast-forward to see the final result.
    Refer to the exercises on the **Introduction** page for guidance on what to look for.
    """
)

# --- A. TOP PANEL: Controls in horizontal layout ---
config_tab1 = parameters_1d("tab1")

# Reset button in its own row
col_reset, col_spacer = st.columns([1, 3])
with col_reset:
    if st.button(
        "Reset / Initialize",
        type="primary",
        key="tab1_reset",
        help="After changing any of the settings / parameters above, click the button to reset the model and start training.",
    ):
        init_session_state(config_tab1)
        st.rerun()

st.markdown("---")

# Initialize on first load
if "tab1_q_table" not in st.session_state:
    init_session_state(config_tab1)

display_state_1d = get_display_state(config_tab1)

# --- B. GRID ---
ready = display_state_1d.get("ready_for_episode", True)
is_terminal = display_state_1d.get("is_terminal", True)
episode_completed_via_step = display_state_1d.get(
    "episode_completed_via_step", False
)
show_dog = (
    (config_tab1["start_mode"] == "Fixed")
    or (not ready)
    or (ready and is_terminal and episode_completed_via_step)
)
show_final_path = ready and is_terminal and episode_completed_via_step
render_grid_1d(
    config_tab1["start_pos"],
    config_tab1["end_pos"],
    display_state_1d["current_state"],
    config_tab1["goal_pos"],
    display_state_1d["current_path"],
    show_path=not ready,
    show_dog=show_dog,
    show_final_path=show_final_path,
)
# --- C. TRAINING CONTROLS (full width) ---
render_training_controls(
    config_tab1,
    display_state_1d,
    reset_episode,
    step_agent,
    run_batch_training,
)

# --- D. Q-MATRIX (full width) ---
with st.expander("Current Q-Matrix", expanded=True):
    st.dataframe(
        display_state_1d["q_table"].style.highlight_max(
            axis=1, color="lightgreen"
        ),
        width="stretch",
    )

# --- E. OPTIMAL ACTIONS (full width) ---
with st.expander("Optimal Actions", expanded=True):
    render_policy_1d(
        display_state_1d["q_table"],
        config_tab1["start_pos"],
        config_tab1["end_pos"],
        config_tab1["goal_pos"],
    )

# --- F. CHARTS (full width, separate rows) ---
with st.expander("Evolving Q-Values", expanded=False):
    render_q_history_chart(display_state_1d["q_history_plot"])

with st.expander("Steps per Episode", expanded=False):
    render_steps_chart(
        display_state_1d["steps_per_episode"],
        display_state_1d["q_history_plot"],
    )

autoplay_pending_key = f"{config_tab1.get('tab_id', 'tab1')}_autoplay_pending_rerun"
if st.session_state.get(autoplay_pending_key, False):
    st.session_state[autoplay_pending_key] = False
    st.rerun()

# --- 2D Grid tab hidden for now ---
# To re-enable, restore tabs and the tab_2d block from app backup

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: grey; font-size: 0.85em;">'
    "&copy; 2026 Emre Ozdenoren. All rights reserved."
    "</p>",
    unsafe_allow_html=True,
)
