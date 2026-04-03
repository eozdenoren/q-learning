"""Dog & Bone Q-learning demo page with Environment and Exercises tabs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
from streamlit_app.ui.controls import parameters_1d
from streamlit_app.ui.training import render_training_controls
from streamlit_app.ui.grid import render_grid_1d
from streamlit_app.ui.charts import render_q_history_chart
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

tab_env, tab_exercises = st.tabs(["The Environment", "Exercises"])

# ==============================================================================
# TAB 1: THE ENVIRONMENT
# ==============================================================================
with tab_env:
    st.markdown(
        r"""
### The Setup

Luna the robot dog stands on a corridor — a line of positions from left to right.
Somewhere on the line there is a bone. Luna can move **left** or **right**. That's it.

- When she reaches the bone: **reward = 10**
- Anywhere else: **reward = 0**
- Luna has **no idea where the bone is**

### Luna's Memory: The Q-Table

Luna keeps a table called the **Q-table** with one row per position and one column
per action (Left, Right). Each cell stores her current estimate of how good it is
to take that action from that position. At the start, every entry is **0**.

$Q(s, a)$ = "if I'm in state $s$ and take action $a$, then play optimally, how much total reward can I expect?"

### How Luna Learns

At each step:
1. **Choose** an action — usually the best in the Q-table (exploit), sometimes random (explore)
2. **Observe** the reward and the new state
3. **Update** one cell of the Q-table:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

The term in brackets is the **surprise** — the difference between what happened and what Luna expected.
She nudges the Q-value by a fraction $\alpha$ of this surprise.

### The Parameters

| Parameter | What it does |
|-----------|-------------|
| $\alpha$ (learning rate) | How much Luna adjusts toward new information. High = learn fast. |
| $\gamma$ (discount factor) | How much future rewards matter. High = patient, low = myopic. |
| $\varepsilon$ (exploration rate) | How often Luna tries a random action instead of the best one. |

No one tells Luna where the bone is. No one programs a strategy. She has only
a reward signal and an update rule. **The strategy emerges from experience.**
        """
    )

    with st.expander("Aside: Q-Learning and the Human Brain"):
        st.markdown(
            r"""
Q-learning is closely related to how animals actually learn.

**Dopamine and the surprise signal.** In the 1990s, neuroscientist Wolfram Schultz
discovered that dopamine neurons fire in a pattern that matches the "surprise" term
in the Bellman update. Unexpected reward → large dopamine burst (positive surprise).
Expected reward → no response. Omitted expected reward → dopamine dip (negative surprise).
After learning, dopamine shifts from the reward to earlier predictive cues — exactly
how Q-values propagate backward through Luna's table.

| Q-Learning | The Brain |
|---|---|
| Reward $r$ (10 at the bone) | Primary rewards (food, warmth) |
| Surprise $[r + \gamma \max Q - Q]$ | Phasic dopamine — drives learning |
| Q-values guide action selection | Learned values guide behaviour via "wanting" |

**The punchline:** Q-learning was developed by mathematicians independently of
neuroscience. The convergence was discovered after the fact — computer scientists
and neuroscientists had arrived at the same learning mechanism from entirely
different directions.
            """
        )


# ==============================================================================
# TAB 2: EXERCISES
# ==============================================================================
with tab_exercises:
    st.markdown(
        """
### Instructions

Use the simulation below to complete the exercises. For each exercise, record your
answer — you will be asked to share your findings with the class.
        """
    )

    # --- Controls ---
    config_tab1 = parameters_1d("tab1")

    col_reset, col_spacer = st.columns([1, 3])
    with col_reset:
        if st.button(
            "Reset / Initialize",
            type="primary",
            key="tab1_reset",
            help="Click after changing parameters to reset and start fresh.",
        ):
            init_session_state(config_tab1)
            st.rerun()

    st.markdown("---")

    if "tab1_q_table" not in st.session_state:
        init_session_state(config_tab1)

    display_state_1d = get_display_state(config_tab1)

    # --- Grid ---
    ready = display_state_1d.get("ready_for_episode", True)
    is_terminal = display_state_1d.get("is_terminal", True)
    episode_completed_via_step = display_state_1d.get("episode_completed_via_step", False)
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

    # --- Training controls ---
    render_training_controls(
        config_tab1,
        display_state_1d,
        reset_episode,
        step_agent,
        run_batch_training,
    )

    # --- Q-Matrix ---
    with st.expander("Current Q-Matrix", expanded=True):
        st.dataframe(
            display_state_1d["q_table"].style.highlight_max(axis=1, color="lightgreen"),
            width="stretch",
        )

    # --- Optimal Actions ---
    with st.expander("Optimal Actions", expanded=True):
        render_policy_1d(
            display_state_1d["q_table"],
            config_tab1["start_pos"],
            config_tab1["end_pos"],
            config_tab1["goal_pos"],
        )

    # --- Q-Value history ---
    with st.expander("Evolving Q-Values", expanded=False):
        render_q_history_chart(display_state_1d["q_history_plot"])

    autoplay_pending_key = f"{config_tab1.get('tab_id', 'tab1')}_autoplay_pending_rerun"
    if st.session_state.get(autoplay_pending_key, False):
        st.session_state[autoplay_pending_key] = False
        st.rerun()

    # --- Exercises ---
    st.markdown("---")
    st.markdown(
        r"""
### Exercise 1: Watch Learning Spread

1. Click **"Train a new episode step by step"** and then click **"Take Next Step"** repeatedly to watch Luna's first episode.
2. After Luna finds the bone, look at the **Q-Matrix**.
3. **Write down:** How many cells are non-zero after episode 1? After episode 2? After episode 3?
4. **Explain:** Why does learning spread outward from the bone by roughly one position per episode?

---

### Exercise 2: Are Wrong Moves Worthless?

1. Use **Fast Forward** to train for 20 episodes.
2. Pick a position that is 3 steps away from the bone.
3. **Write down:** the Q-value for moving *toward* the bone and the Q-value for moving *away*.
4. **Answer:** Is the "wrong direction" Q-value zero or positive? Why?

> *Hint: if Luna moves away from the bone, can she still eventually reach it?
> What does that imply about the discounted reward?*

---

### Exercise 3: The Discount Factor

1. **Reset** and set $\gamma = 0.9$. Train for 20 episodes. Write down Q(position 0, Right).
2. **Reset** and set $\gamma = 0.5$. Train for 20 episodes. Write down Q(position 0, Right).
3. **Answer:** Which $\gamma$ gives higher Q-values far from the bone? What does this mean about how "patient" the agent is?

---

### Exercise 4: From Dogs to Firms

**Think about this before moving to the Algorithmic Pricing tab:**

Luna has one goal (find the bone) and learns alone. Now imagine *two* agents,
each trying to maximise its own reward, where one agent's actions affect the
other's reward.

**Write down:** What do you think will happen? Will they compete or cooperate?
Does your answer depend on $\gamma$?
        """
    )

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: grey; font-size: 0.85em;">'
    "&copy; 2026 Emre Ozdenoren. All rights reserved."
    "</p>",
    unsafe_allow_html=True,
)
