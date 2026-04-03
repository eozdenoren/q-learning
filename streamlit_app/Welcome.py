"""Home page for Q-Learning demo multipage Streamlit app."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Q-Learning App", layout="wide")

st.title("Q-Learning: From Robot Dogs to Algorithmic Pricing")

st.markdown(
    """
This app contains two interactive simulations that teach you how Q-learning works
and what happens when firms use it to set prices.

---

### 1. Dog & Bone

A robot dog named **Luna** stands on a corridor. Somewhere on the corridor there is a bone.
Luna can move left or right. She gets a reward when she reaches the bone — but
**she has no idea where the bone is.** She starts as a complete blank slate.

Using Q-learning — a simple trial-and-error algorithm — Luna learns to find the bone
from any starting position. No one programs a strategy. It **emerges** from experience.

**Go to the Dog & Bone tab** to watch Luna learn step by step, inspect her Q-table,
and experiment with the learning parameters.

---

### 2. Algorithmic Price Competition

Now replace the dog with **two gas stations** on a main road, each using its own
Q-learning algorithm to set fuel prices. Each station observes its own profit and
adjusts. There is no communication between the stations, no shared data, no intent
to coordinate.

**The question:** do the algorithms learn to compete (Nash equilibrium) or do they
learn to keep prices high (tacit collusion)?

**Go to the Algorithmic Pricing tab** to train competing pricing algorithms and
see what pricing behaviour emerges.

---

*Use the sidebar on the left to navigate between the pages.*
"""
)

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: grey; font-size: 0.85em;">'
    "&copy; 2026 Emre Ozdenoren. All rights reserved."
    "</p>",
    unsafe_allow_html=True,
)
