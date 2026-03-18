# Q-Learning Case Study: Teaching Notes

## Overview

This case study uses an interactive web app to teach Q-learning and illustrate how
algorithmic pricing can lead to tacit collusion. It proceeds in three stages:

1. **Luna the robot dog** — builds intuition for Q-learning via a simple bone-finding task
2. **The formalism** — gives students the Bellman equation and parameters
3. **Pricing competition** — applies the same algorithm to a duopoly pricing game

The key pedagogical arc: *a simple learning rule + rewards can produce sophisticated
emergent strategies that no one explicitly programmed — including collusion.*

---

## Part 1: Luna and the Bone (Introduction + Dog & Bone tab)

### What students should discover through experimentation

**1. Learning propagates backward from the reward ("the wave").**
- After Episode 1, only states adjacent to the bone have non-zero Q-values.
- After Episode 2, states two steps away also get updated.
- After Episode N, the learning has spread roughly N steps from the bone.
- *Insight:* Learning is local and slow. The agent only learns about states it has
  actually visited, and knowledge propagates one step at a time through the Bellman
  update's max Q(s',a') term. This is fundamentally different from supervised learning.

**2. Wrong moves have positive Q-values.**
- Students often expect Q(s, "move away from bone") = 0. It's not.
- The Q-value represents: "if I take this action and then act optimally, how much
  discounted reward will I eventually get?"
- Even after a wrong step, the agent can still reach the bone — it just takes longer,
  so the discounting reduces the value.
- *Insight:* Q-values are not "good vs. bad" labels. They are predictions of future
  cumulative reward. Every action that doesn't make the goal permanently unreachable
  has some positive value. The agent chooses by comparing relative Q-values, not by
  checking if a Q-value is zero.

**3. The discount factor gamma controls the "steepness" of the Q-value landscape.**
- High gamma (close to 1): Q-values are similar across states. The agent knows the
  bone is reachable from everywhere but has weak preferences about direction.
- Low gamma: Only states very close to the bone have significant Q-values. The agent
  strongly prefers being close to the bone.
- *Insight:* gamma determines how far into the future the agent "cares." This directly
  affects how strongly the agent differentiates between actions.

**4. Emergence: no one programs the strategy.**
- Luna has no built-in notion of "walk toward the bone." She has only:
  (a) a table of zeros, (b) a reward signal, (c) an update rule.
- Yet a coherent, optimal strategy emerges from experience.
- This is the central lesson that transfers to the pricing setting.

### Teaching suggestions for Part 1

- Have students run step-by-step through the first 2-3 episodes to see the wave.
- Ask them to predict what the Q-table will look like before running the next episode.
- After the wave insight, have them fast-forward to convergence and examine the final
  Q-table. Ask: "Why isn't Q(s, wrong direction) = 0?"
- The 2D grid is useful for showing that the same principles apply in more complex
  state spaces, and that multiple optimal paths can exist.

---

## The Neuroscience Connection (for classroom discussion)

The Introduction page has a collapsible aside on the Q-learning / brain parallel. Below
is a fuller version for classroom use, especially if students ask deep questions.

### The core parallel

Q-learning's surprise term [r + gamma * max Q(s',a') - Q(s,a)] maps onto dopamine
prediction errors discovered by Wolfram Schultz (1990s). Key findings:

- Unexpected reward → large dopamine burst (positive surprise)
- Expected reward → no dopamine response (zero surprise)
- Omitted expected reward → dopamine dip below baseline (negative surprise)
- After learning, dopamine shifts from reward to earlier predictive cues — just as
  Q-values propagate backward from the bone to earlier states

### "Why does the organism follow its Q-values?"

This question will likely come up. There are three levels:

**Robots (Luna):** Trivial — we wrote `argmax Q` in the code. No motivation, no
experience, no mystery.

**Simple organisms:** Evolution built nervous systems where higher learned values
generate stronger motor drive. This is Berridge's "wanting" system — dopamine-mediated
motivational pull that translates learned values into approach behavior. The animal
doesn't "decide" to follow its Q-values; it is drawn toward high-value actions by its
neural circuitry, the way gravity pulls a ball downhill. No reasoning, planning, or
understanding of the future is required. The future is already encoded in the Q-values
by the learning process; acting on them is automatic.

**Humans:** We have the same value-following machinery. Most daily behavior (reaching
for your phone, taking the usual route to work, opening the fridge) is driven by
learned values without conscious deliberation. The difference is that humans can
sometimes *override* the system — willpower, delayed gratification, executive control
via the prefrontal cortex. But overriding takes effort, and we often fail (think of
diets, procrastination, addiction). This is exactly what you'd expect if the default
mode is automatic value-following with an imperfect override switch.

**The deep question:** Where does automatic drive end and conscious choice begin? Is
"free will" a real override or a subjective illusion layered on top of the same
value-following machinery? This is one of the deepest open questions in philosophy of
mind and neuroscience. You don't need to resolve it for this case — but it's a
fascinating discussion if students want to go there.

### Why this matters for the case

The key takeaway for students: **you do not need to assume that agents are rational,
strategic, or even conscious for sophisticated behavior to emerge.** All you need is:

1. A reward signal
2. A learning rule (the Bellman update)
3. A value-following mechanism (pick the action with the highest Q-value)

In the bone example, this produces an optimal search strategy. In the pricing example,
this produces tacit collusion. Neither strategy was programmed, intended, or understood
by the agents. This is what makes algorithmic collusion a genuine policy concern: it
can arise without intent, communication, or even awareness.

### Tangential topics to be ready for

- **"Wanting" vs "liking" (Berridge):** Dopamine ≈ wanting (motivational pull), not
  liking (hedonic pleasure, which is opioid-mediated). Dopamine-depleted rats enjoy
  sugar on their tongue but won't walk to get it. In Q-learning terms: they experience
  the reward r but can no longer use Q-values to select actions.

- **Off-policy vs on-policy:** Q-learning is off-policy (learns value of optimal path
  regardless of what agent actually does). Some neuroscientists argue the brain looks
  more like SARSA (on-policy — learns value of the path actually taken). For general
  understanding this distinction is minor, but worth knowing if a sharp student asks.

- **Model-free vs model-based:** Q-learning is model-free (no internal model of the
  environment). Humans clearly also do model-based reasoning ("if I take this route,
  I'll hit traffic"). The brain likely uses both systems. For this case, model-free
  is the relevant one.

---

## Part 2: The Formalism

### Key points to emphasize

- The Bellman equation is just a formalization of what students already observed:
  the Q-value of an action depends on the immediate reward plus the best future
  Q-value from the next state.
- alpha controls how quickly old estimates are overwritten (learning rate).
- gamma is the discount factor they already experimented with.
- epsilon controls the explore/exploit tradeoff.
- Updates happen after every step, not every episode. This is why learning is gradual.

### Common student confusions

- "Why not just set alpha = 1?" → Because the environment involves randomness
  (epsilon-greedy exploration). Averaging over multiple experiences gives more
  stable estimates.
- "Why explore at all?" → Without exploration, the agent might never discover
  that a different action is better. It could get stuck exploiting a suboptimal
  action forever.
- "Why discount?" → Without discounting (gamma = 1), the agent would be indifferent
  between reaching the bone in 1 step and reaching it in 1000 steps. Discounting
  creates a preference for shorter paths.

---

## Part 3: Pricing Competition (Economics Pricing Example tab)

### The setup

- Two firms (Alice and Bob) simultaneously choose prices from a discrete set.
- Demand: q_i = k1 - p_i + k2 * p_j (linear, with substitution parameter k2).
- Profit: pi_i = (p_i - c) * q_i.
- Each firm uses independent Q-learning with profits as rewards.
- State = (p1, p2) from the previous period. Action = own price for this period.

### Key economic concepts

- **Nash equilibrium price (p_e):** The static best-response equilibrium. Each firm
  prices at p_e = (k1 + c) / (2 - k2).
- **Collusive price (p_c):** The joint-profit-maximizing price. Higher than p_e.
  p_c = (2k1 + 2c(1-k2)) / (4(1-k2)).
- **The question:** Do Q-learning agents converge to p_e (competitive) or p_c (collusive)
  or something in between?

### What students should discover

- With default parameters, agents typically converge to prices *above* the Nash
  equilibrium, often close to the collusive price.
- This is "tacit collusion" — no communication, no explicit agreement, yet both
  agents learn to sustain supra-competitive prices.
- The mechanism: the agents learn that undercutting triggers retaliation (the other
  agent eventually lowers its price too), reducing long-run profits. The discount
  factor gamma is crucial — it makes agents care about these future consequences.
- The normalized profit metric Delta = (avg_profit - pi_e) / (pi_c - pi_e) measures
  how close to full collusion the agents get. Delta = 0 means Nash; Delta = 1 means
  full collusion.

### Connection back to Luna

- The same "emergence" principle: no one programs collusion. The agents have only
  Q-tables, profit signals, and the Bellman update. Yet a sophisticated pricing
  strategy emerges.
- The "wave of learning" from Part 1 has an analog: agents gradually learn the
  consequences of price deviations across multiple periods.
- The discount factor gamma plays the same role: it determines how much agents care
  about future consequences of today's pricing decision.

### Discussion questions for class

1. Is this really "collusion" if no one programmed it and the agents never communicate?
2. Should regulators be concerned about algorithmic pricing? What if real firms deploy
   Q-learning (or similar RL) algorithms to set prices?
3. What happens if one firm uses Q-learning and the other uses a fixed pricing rule?
4. How does the number of price points (m) affect the outcome?
5. What role does the exploration rate play? What happens with no exploration?

---

## Possible extensions

- [ ] Add a "deviation experiment" where students can manually override one firm's
      price after convergence and watch the other firm's response (punishment/retaliation)
- [ ] Compare Q-learning collusion with explicit cartel pricing
- [ ] Explore asymmetric settings (different costs, different learning rates)
- [ ] Connect to real-world cases of algorithmic pricing (airlines, gasoline, online retail)
- [ ] Add a "tournament" mode where students train agents and compete (Pricing Battle tab)

---

## References

- Mitchell, M. (2019). *Artificial Intelligence: A Guide for Thinking Humans.* Chapter 8.
- Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). "Artificial
  Intelligence, Algorithmic Pricing, and Collusion." *American Economic Review.*
