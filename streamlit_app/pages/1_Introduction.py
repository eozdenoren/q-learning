import streamlit as st

st.set_page_config(page_title="Introduction to Q-Learning", layout="wide")

st.title("Introduction to Q-Learning")

st.markdown(
    r"""
### Teaching a Robot Dog to Find a Bone

Imagine a robot dog named Luna standing on a long narrow corridor — a line of positions
stretching from left to right. Somewhere on this line there is a bone. Luna can take
one step to the left or one step to the right. That's it — those are her only moves.
If she steps out of bounds, she simply stays put.

Luna's goal is to reach the bone. When she does, she receives a **reward** — say, 10 points
added to her memory. At any other position, she gets nothing. But here's the catch:
**Luna has no idea where the bone is.** She doesn't know that moving right is better than
moving left, or how far away the bone is. She starts as a complete blank slate.

### Luna's Memory: The Q-Table

How does Luna remember what she's learned? She keeps a table in her memory called the
**Q-table**. It has one row for each position on the line and one column for each action
(Left, Right). Each cell stores a number — Luna's current estimate of how good it is to
take that action from that position. At the start, every entry is 0, because Luna knows
nothing.

### How Luna Learns

At each step, Luna looks at her current position (her **state**) and chooses an action.
Sometimes she picks randomly to explore; other times she picks the action with the
highest Q-value in her table to exploit what she's already learned. After she acts, she
may receive a reward, and she updates a single cell in her Q-table based on what happened.

One **episode** consists of Luna moving around until she reaches the bone. Then she is
placed at a new starting position and a new episode begins.

That's it. No one tells Luna where the bone is. No one programs a strategy. Luna has
only a simple update rule and her Q-table. The question is: **is this enough for Luna to
learn to find the bone from anywhere?**

---

### The Formalism

Let's put precise notation on what Luna is doing.

**States, actions, and rewards.** Luna's **state** $s$ is her current position on the line.
Her **actions** are $\{L, R\}$ (move left or right). She receives a reward $r = 10$ when she
reaches the bone, and $r = 0$ otherwise.

**The Q-table.** $Q(s, a)$ stores Luna's current estimate of how much reward she
expects to eventually collect if she takes action $a$ in state $s$ and then acts optimally
thereafter. Future rewards are **discounted** — a reward received many steps from now is
worth less than the same reward received immediately, just as a pound tomorrow is worth
less than a pound today. The discount factor $\gamma$ controls how much less (more on this
below). Initially, all entries are 0.

**Choosing actions: the $\varepsilon$-greedy policy.** At each step, Luna flips a biased coin:
- With probability $\varepsilon$, she **explores** — picks a random action.
- With probability $1 - \varepsilon$, she **exploits** — picks the action with the highest Q-value.

When $\varepsilon = 1$, Luna acts completely randomly. When $\varepsilon = 0$, she always
follows the Q-table.

**The Bellman update.** After taking action $a$ in state $s$, Luna arrives in a new state
$s'$ and receives a reward $r$ (either 10 if she reached the bone, or 0 otherwise). She
then updates the Q-value for the state-action pair she just tried. The update rule is:

$$Q(s, a) \leftarrow (1 - \alpha)\; \underbrace{Q(s, a)}_{\text{old estimate}} + \alpha \; \underbrace{\left[ r + \gamma \max_{a'} Q(s', a') \right]}_{\text{new estimate}}$$

Here is the logic behind this formula. Luna wants $Q(s, a)$ to reflect the total
discounted reward she can expect after taking action $a$ in state $s$. She has two
sources of information:

- **The old estimate:** the value $Q(s, a)$ currently sitting in her table, based on
  all her past experience.
- **The new estimate:** $r + \gamma \max_{a'} Q(s', a')$, based on what just happened. This
  has two parts:
    1. **The immediate reward** $r$ — what she just received (10 or 0).
    2. **The best future reward from** $s'$ — if Luna plays optimally from her new position,
       the best she can expect is $\max_{a'} Q(s', a')$. But that future reward is one step
       away, so it is discounted by $\gamma$.

Luna blends these two sources using the **learning rate** $\alpha$. When $\alpha$ is small, she
mostly trusts her accumulated experience and adjusts slowly. When $\alpha = 1$, she
fully replaces her old estimate with the new one.

Rearranging, this is equivalent to the form you may see in textbooks:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

In this version, the term in brackets is the **surprise** — the difference between the new
estimate and the old one. Luna nudges her Q-value by a fraction $\alpha$ of this surprise.
    """
)

with st.expander("Aside: Q-Learning and the Human Brain"):
    st.markdown(
        r"""
Q-learning may look like a mathematical abstraction, but it turns out to be closely
related to how humans and animals actually learn.

**Dopamine and the surprise signal.** In the 1990s, neuroscientist Wolfram Schultz
discovered that dopamine neurons in the brain fire in a pattern that looks remarkably
like the "surprise" term in the Bellman update. When a monkey receives an
**unexpected** reward (say, a drop of juice), dopamine neurons fire strongly — a large
positive surprise, just like Luna's first time finding the bone. But after the monkey
has learned that a cue (a light or sound) predicts the reward, something shifts:
dopamine no longer fires at the reward itself (no surprise — it was expected), and
instead fires **at the cue**. The surprise signal has moved backward, from the reward
to the earlier state that predicts it — exactly the way Q-values propagate backward
through Luna's table.

If the expected reward is **omitted**, dopamine drops *below* baseline — a negative
surprise, just like a negative TD error in Q-learning.

**Three things in Q-learning, three things in the brain.** The parallel runs deep:

| Q-Learning | The Brain |
|---|---|
| Reward $r$ (10 at the bone) | Primary rewards (food, warmth, etc.) |
| Surprise $[r + \gamma \max Q - Q]$ | Phasic dopamine (bursts) — drives learning |
| Q-values guide action selection | Learned values guide behaviour via "wanting" |

**"Wanting" vs. "liking."** You might assume that the pleasure of eating food is
dopamine. But Kent Berridge showed in the 1990s that dopamine is really about
**wanting** — the motivational pull toward something — not **liking** (the hedonic
pleasure, which involves opioid systems). Rats with dopamine depleted still *enjoy*
sugar placed on their tongue but won't *walk across the cage* to get it. In Q-learning
terms: they still experience the reward $r$, but they can no longer use their Q-values
to select actions.

**After learning is complete**, there are no more surprises — the Q-table is accurate,
so every update is essentially zero. Dopamine bursts go quiet. But Luna still walks
toward the bone, because **acting and learning are separate**. Luna acts by reading her
Q-table and picking the best action. She learns by updating the table when surprised.
Once the table is complete, learning stops but acting continues. In the brain, tonic
(baseline) dopamine and the learned value representations continue to drive motivated
behaviour even after the phasic surprise signal has nothing left to teach.

**The punchline:** Q-learning was developed by mathematicians independently of the
neuroscience. When Schultz discovered that dopamine neurons encode prediction errors,
it was a striking convergence — computer scientists and neuroscientists had arrived at
the same learning mechanism from entirely different directions.
        """
    )

st.markdown(
    r"""
To summarise the parameters:
- $\alpha$ (**learning rate**, between 0 and 1): How much Luna adjusts toward the new estimate.
- $\gamma$ (**discount factor**, between 0 and 1): How much a reward one step in the future is worth compared to a reward right now. A reward $k$ steps in the future is worth $\gamma^k$ times its face value.
- $r$: the immediate reward (10 at the bone, 0 elsewhere).
- $\max_{a'} Q(s', a')$: the best Q-value available at the next state — this is how Luna "looks ahead" using what she has already learned.

Note that Luna updates her Q-table **after every single step**, not just at the end of an
episode. This is crucial. The only time Luna receives an actual reward ($r = 10$) is when
she reaches the bone. But at every other step, the new estimate
$r + \gamma \max_{a'} Q(s', a')$ can still be positive — because $\max_{a'} Q(s', a')$ reflects
how valuable the next state is based on what Luna has already learned. If that new
estimate exceeds Luna's old estimate $Q(s,a)$, there is a positive surprise, and the
Q-value gets nudged upward. This is how learning propagates: once the states near the
bone have high Q-values, arriving at those states from one step further away generates
a surprise, which updates the Q-values one step further out — and so on, spreading
gradually through the table.

---

### Exercises: The 1D Grid

Head over to the **Dog & Bone Example** tab and select the **1D Grid**. Use step-by-step
mode to watch Luna learn.

**Exercise 1: How does learning spread?**
Run through the first episode step by step. After Luna finds the bone for the first
time, look at the Q-table.
- How many cells are non-zero?
- Now run a second episode. How many new cells became non-zero?
- What about after the third episode?
- You should see a pattern: learning spreads outward from the bone by roughly one step
  per episode. Why does this happen? *(Hint: look at the Bellman update — what determines
  whether a Q-value changes from zero to something positive?)*

**Exercise 2: Are wrong moves worthless?**
Train for 10 or more episodes (you can use fast-forward). Now pick a state that is,
say, 3 positions away from the bone.
- What is the Q-value for moving *toward* the bone?
- What is the Q-value for moving *away* from the bone?
- Is the "wrong direction" Q-value zero, or positive?
- If it's positive, explain why. *(Hint: if Luna moves away from the bone, is she stuck
  forever? Or can she still reach the bone — just in more steps? What does that imply
  about the discounted reward she can expect?)*

**Exercise 3: What does the discount factor do?**
Reset the model and set $\gamma = 0.9$. Train until the Q-table stabilises. Write down the
Q-values for a state far from the bone. Now reset and repeat with $\gamma = 0.5$.
- With high $\gamma$, are the Q-values far from the bone large or small relative to those
  near the bone?
- With low $\gamma$, how do they compare?
- What does this tell you? *(A high $\gamma$ means Luna values distant rewards almost as
  much as nearby ones. A low $\gamma$ means she is "impatient" — only states close to the
  bone matter much.)*

---

### Exercises: The 2D Grid

Now switch to the **2D Grid** tab and train until convergence. Look at the **Optimal
Actions** plot, which shows arrows indicating the best action(s) at each position.

**Exercise 4: When are two actions equally good?**
Find a state where the bone is diagonally away — say, 2 steps right and 3 steps down.
- The plot should show a diagonal arrow, meaning Luna is **indifferent** between two
  actions (e.g., Right and Down).
- Why are these two actions equally good? *(Hint: if Luna goes Right, how many steps
  remain on the shortest path to the bone? If she goes Down instead, how many steps
  remain? What does this imply about the discounted reward from each action?)*
- Now look at a state directly to the left of the bone (same row). Is there still
  indifference, or does one action dominate? Why?

**Exercise 5: The Q-table knows the whole map.**
Look at the full Q-table after convergence. Luna has learned a Q-value for every single
state-action pair — not just the states she visits on any one trip, but *all* of them.
Through many episodes starting from random positions, she has explored the entire grid.
- How many entries does the Q-table have? *(Count: number of grid positions × number of
  actions.)*
- In the 1D case with 11 positions, the Q-table had $11 \times 2 = 22$ entries.
  On a 5×5 grid, it has $25 \times 4 = 100$ entries. What if the grid were 100×100?
  What about 1000×1000?

This is the fundamental limitation of the Q-table approach: **the table grows
exponentially with the complexity of the environment.** A 100×100 grid requires 40,000
entries. A problem with 10 dimensions, each with 100 possible values, would require
$100^{10} = 10^{20}$ entries — more than the number of grains of sand on Earth.

In practice, most interesting problems (self-driving cars, game-playing, real-world
pricing with many products) have state spaces far too large for a table. The solution
is to replace the Q-table with a **neural network** that takes the state as input and
outputs Q-value estimates for each action. This is called **Deep Q-Learning** (DQN). The
network *generalises* — it learns patterns across states, so it can estimate Q-values
even for states it has never visited. The trade-off is that it approximates rather than
memorises: it's scalable but no longer exact.

For our pricing exercises later, the environment is small enough for a Q-table. But keep
this limitation in mind — in richer settings, neural networks become necessary, and
understanding what strategies they learn becomes much harder.

---

### From Dogs to Firms

Luna is not programmed with a strategy — she has only a simple update rule and a reward.
Yet a coherent strategy *emerges* from her experience. No one told Luna to walk toward the
bone; she figured it out on her own.

Now consider this: what if instead of one agent searching for a bone, we had *two* gas
stations setting fuel prices on a main road? Each station uses the same Q-learning
algorithm, receiving profits as its reward. **What kind of pricing strategies might
emerge — and would they be the strategies that economists expect?**

Head to the **Economics Pricing Example** tab to find out.
    """
)

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: grey; font-size: 0.85em;">'
    "&copy; 2026 Emre Ozdenoren. All rights reserved."
    "</p>",
    unsafe_allow_html=True,
)
