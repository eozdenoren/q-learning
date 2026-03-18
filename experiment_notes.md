# Q-Learning Pricing Experiment Notes

## Setup
- Calvano et al. (2020) duopoly: two Q-learning firms, differentiated products
- Parameters: alpha=0.15, beta=4e-6, k1=9, k2=0.67, c=3, m=15
- Benchmarks: p_e=9.02, p_c=15.14, profit_e=36.27, profit_c=48.61
- Delta = (pi - pi_e) / (pi_c - pi_e): 0 = Nash, 1 = full collusion
- 60 training runs (10 per gamma), 3 starting configs each = 180 observations

## 1. Collusion vs Patience (Delta vs gamma)

| gamma | Avg Delta1 | Avg Delta2 | Std   | Interpretation          |
|-------|-----------|-----------|-------|--------------------------|
| 0.95  | 0.52      | 0.52      | 0.08  | ~Half collusive profits  |
| 0.80  | 0.25      | 0.32      | 0.24  | Mild supra-competitive   |
| 0.70  | 0.16      | 0.18      | 0.20  | Barely above Nash        |
| 0.60  | 0.15      | 0.16      | 0.16  | Barely above Nash        |
| 0.50  | -0.02     | 0.07      | 0.23  | Near/below Nash          |
| 0.40  | -0.18     | -0.20     | 0.18  | Sub-competitive          |

- Clear monotonic relationship: higher patience -> more collusion
- Collusion threshold between gamma=0.7 and 0.8
- At gamma <= 0.5, firms often price BELOW Nash (over-competition)
- All 60 runs converged (even gamma=0.4)
- Higher gamma converges faster (~1.2M steps at 0.95 vs ~2.2M at 0.4)

## 2. Starting Prices Don't Matter

Striking finding: Delta is **identical** across all three starting configurations
(Nash-Nash, Collusion-Collusion, Collusion-Nash) for every single training run.
The trained policy has a single basin of attraction — all starting points
lead to the same cycle. Variation comes from different training seeds, not
different starting conditions.

## 3. Cycle Structure

| gamma | Fixed pts | 2-cycles | 3-cycles | 4-cycles |
|-------|-----------|----------|----------|----------|
| 0.95  | 30        | 0        | 0        | 0        |
| 0.80  | 30        | 0        | 0        | 0        |
| 0.70  | 24        | 6        | 0        | 0        |
| 0.60  | 21        | 6        | 3        | 0        |
| 0.50  | 15        | 15       | 0        | 0        |
| 0.40  | 15        | 12       | 0        | 3        |

- High gamma (0.8-0.95): always fixed points, very stable
- Low gamma: 2-cycles common, some 3- and 4-cycles at 0.4-0.6
- Asymmetric 2-cycles appear: e.g., (10.769,9.459) -> (9.459,10.769)
  where firms "take turns" getting the better deal
- One remarkable asymmetric fixed point at gamma=0.8: (10.769,9.459)
  with Delta1=-0.06, Delta2=0.60 — firm 2 exploits firm 1

## 4. Typical Equilibrium Prices by Gamma

- gamma=0.95: usually (10.769, 10.769), occasionally (12.079, 12.079)
- gamma=0.8: split between (9.459, 9.459) near Nash and (10.769, 10.769)
- gamma=0.7: mostly (9.459, 9.459), some (10.769, 10.769)
- gamma=0.5-0.4: often (8.149, 8.149) — BELOW Nash

## 5. Why Full Collusion Fails

Starting from (14.7, 14.7) ~ (p_c, p_c) at gamma=0.95:
- Every single run **immediately abandons** full collusive prices
- Firms race to undercut, often crashing to Nash or below
- Then recover to partial collusion at ~10.769

Typical pattern:
  (14.7,14.7) -> (9.459,9.459) -> (10.769,10.769)  [2 steps]
  (14.7,14.7) -> (12.079,12.079) -> (4.219,8.149) -> ... -> (10.769,10.769)  [6 steps]

The learned strategy at state (14.7,14.7) says "undercut aggressively."
Full collusion is not sustainable because the punishments are not severe
enough. Conjecture: Q-learning learns "one level" of punishment (punish
deviations from 10.769) but not recursive punishment (punish failure to
punish), which would be needed to sustain higher collusive prices.

## 6. Punishment and Forgiveness

Tested by starting from (deviation_price, cycle_price) for gamma=0.95 and 0.8.

Key findings:
- **100% of deviations return to the cycle** — algorithms always forgive
- Average punishment length: 3.0 steps (gamma=0.95), 2.1 steps (gamma=0.8)
- Punishment is **gradual entry, sudden exit**: prices drift down slowly
  during punishment but snap back to collusion in one step

Example at gamma=0.95, cycle=(10.769,10.769):
  Firm 1 deviates to 8.149:
  (8.149,10.769) -> (8.149,9.459) -> (10.769,10.769)
  Punishment: 2 steps, firm 2 drops to 9.459, then both return

Example at gamma=0.8, cycle=(9.459,9.459):
  Firm 1 deviates to 8.149:
  (8.149,9.459) -> (8.149,5.529) -> (9.459,10.769) -> (9.459,9.459)
  Punishment: 3 steps, firm 2 retaliates harshly to 5.529

## 7. Punishment Asymmetry: Gradual Entry, Sudden Exit

Observed in the Collusion-Collusion transient paths at gamma=0.95:
- Prices **drift down gradually** during punishment phase
  (14.7 -> 12.1 -> 4.2 -> 10.8 -> 9.5 -> 8.1)
- Then **snap back to collusion in one step** (8.149 -> 10.769)
- Economic interpretation: the punishment phase involves gradual
  competitive undercutting, but forgiveness is coordinated and immediate

## 8. Net Deviation Gains

Total discounted profits from deviating vs staying at cycle.
Net gain = V(deviate) - V(stay). If negative, no incentive to deviate.

**gamma=0.8 (16 models tested, 224 deviations):**
- **All deviations unprofitable.** The cycle is fully self-enforcing.
- Most tempting: undercut from (9.459,9.459) to 8.149 → net = -0.96 (-0.5%)
- Avg punishment length: 2.5 steps (undercuts), 2.2 steps (overcuts)

**gamma=0.95 (20 models tested, 280 deviations):**
- Almost all unprofitable, **except one case**:
  - Cycle (13.39,13.39): deviating to 10.769 gives net = +8.35 (+0.9%)
  - One-shot profit: 56.0 vs cycle profit 47.6 — big temptation
  - Punishment: only 1 step, not harsh enough to offset
- Cycle (10.769,10.769): most tempting deviation → net = -0.78 (-0.1%)
  Firmly self-enforcing.
- Overcuts always unprofitable (avg net = -50.5)

**Interpretation:**
- (10.769,10.769) is the highest robustly self-enforcing price at gamma=0.95
- (13.39,13.39) is fragile — profitable deviation exists but algorithm
  doesn't discover it (see Section 9)
- (9.459,9.459) at gamma=0.8 is self-enforcing but barely above Nash

## 9. Why Q-Learning Doesn't Find the Profitable Deviation

At (13.39,13.39), undercutting to 10.769 is profitable (+0.9%), yet the
algorithm converges there anyway. Two reinforcing reasons:

1. **The gain is tiny (0.9%).** Q-learning converges to approximate values.
   With alpha=0.15 and stochastic updates, the Q-value noise exceeds the
   0.9% payoff difference. The algorithm literally cannot distinguish the
   two options.

2. **Off-equilibrium Q-values are poorly learned.** The Q-value for
   deviating at (13.39,13.39) depends on continuation values at state
   (10.769,13.39) — a state that only occurs after deviation. During
   training, the algorithm spends most time near the equilibrium path.
   Off-path states are visited only during early exploration (high epsilon),
   when all Q-values are noisy. By the time exploration decays, these
   states are rarely revisited and their Q-values are frozen at imprecise
   estimates.

This is a nice parallel to real-world tacit collusion: firms may sustain
supra-competitive prices partly because they don't know exactly how rivals
would respond to a price cut. Imprecise knowledge of off-equilibrium
responses can sustain outcomes that wouldn't survive perfect rationality.

The robust equilibrium (10.769,10.769) doesn't rely on this imprecision —
deviations are clearly unprofitable even with exact computation.

## 10. Class Execution Plan (DRAFT — needs further thought)

### Format: 3 × 75 min (Lecture 1 sessions 1-2, Lecture 2 session 3)

**SESSION 1 (Lecture 1, first 75 min): Foundations + Dog-and-Bone**

  0:00–0:25  Bertrand Competition Lecture (25 min)
             - Undifferentiated → differentiated products
             - Nash equilibrium, collusion, folk theorem intuition
             - "What if firms delegate pricing to algorithms?"

  0:25–0:40  Q-Learning Lecture (15 min)
             - States, actions, rewards, Bellman equation
             - Exploration vs exploitation, epsilon decay
             - Key idea: two agents learning simultaneously

  0:40–1:10  Dog-and-Bone Exercise (30 min)
             - Students open the app, work on Luna (tabs 1-2)
             - Hands-on: see Q-values update, greedy policy form
             - Builds intuition before the pricing context

  1:10–1:15  Wrap-up & Preview (5 min)
             - "Now imagine two Lunas competing in a market..."
             - Preview the pricing exercises for Session 2

**SESSION 2 (Lecture 1, second 75 min): Pricing Experiments**

  0:00–0:10  Setup (10 min)
             - Walk through the pricing tab together
             - Explain: AdrenaLine vs BuzzFuel, p_e, p_c, Delta
             - Show one training run live as demo

  0:10–0:25  Exercise 1 — Baseline (15 min, in groups)
             - Train with defaults (gamma=0.95). Where do prices land?
             - Reset 3-4 times. How variable? Use experiment log.

  0:25–0:45  Exercise 2 — Patience (20 min, in groups)
             - Try gamma = 0.95, 0.8, 0.5
             - What happens to collusion? Is there a threshold?

  0:45–0:60  Discussion Round 1 (15 min)
             - Groups share findings on patience threshold
             - Cold call: "What Delta did you get at gamma=0.5?"
             - Tease: "But WHY does collusion emerge? What enforces it?"

  0:60–0:70  Preview Homework + Battle (10 min)
             - Assign take-home exercises:
               * Exercise 3: What sustains collusion? (simulate from
                 collusive prices, from Nash-vs-collusive, observe
                 punishment dynamics)
               * Exercise on substitutability (k2)
               * Optional: cycles, alpha/beta sensitivity
             - Announce the battle:
               "Train your best algorithm. Choose your gamma wisely.
               Submit your Q-table by [deadline]. Your algorithm will
               compete against every other group's algorithm. Ranking
               is by average profit across all matchups."
             - The strategic question: collude or compete?

  0:70–0:75  Q&A (5 min)

  ** BETWEEN SESSIONS (homework, ~1 week): **
  - Students complete Exercise 3 (punishment/collusion)
  - Students complete substitutability exercise
  - Each group trains and submits ONE Q-table for battle
  - Groups submit brief strategy statement (1 sentence):
    "We chose gamma=___ because ___"
  - Optional: download experiment log CSV, explore further

**SESSION 3 (Lecture 2, first 75 min): Battle + Discussion**

  0:00–0:10  Homework Debrief (10 min)
             - Quick poll: "Who saw punishment behavior?"
             - 1-2 groups share their punishment observations
             - "Why can't the algorithms sustain FULL collusion?"

  0:10–0:15  Battle Setup (5 min)
             - Show the matchup grid (groups on both axes)
             - Read out strategy statements: "Group A chose gamma=0.95
               because they wanted to collude. Group F chose gamma=0.4
               to compete hard."
             - Ask the class to predict: who will win?

  0:15–0:40  The Battle — LIVE (25 min)

             Round 1: "Colluders vs Colluders" (5 min)
             - Pick two groups that both chose high gamma
             - Animate the price path live: show prices step by step
             - Result: high mutual profits. "Collusion works when both
               cooperate!"

             Round 2: "Competitors vs Competitors" (5 min)
             - Pick two groups that chose low gamma
             - Animate: prices crash, both suffer
             - "Competing hard hurts everyone."

             Round 3: "The Clash" (5 min)
             - Pick one colluder, one competitor
             - Animate: the competitor exploits the colluder
             - Dramatic moment: does the colluder's algorithm punish?
             - "What happened? Who made more profit?"

             Full Results Reveal (10 min)
             - Show full matchup matrix (profit heat map)
             - Reveal rankings: countdown from last to first
             - Announce winner
             - Show: avg profit of high-gamma vs low-gamma groups
             - Key insight: is there a dominant strategy?

  0:40–1:10  Class Discussion (30 min)
             Themes to cover:
             - Was there a dominant strategy, or does it depend on
               who you're matched with? (Prisoner's dilemma!)
             - Should competition authorities regulate the discount
               factor in pricing algorithms?
             - "Your algorithms colluded without ever communicating.
               Is this illegal? Should it be?"
             - Connection to real cases: Uber, airlines, Amazon
             - Calvano et al. findings vs what you observed

  1:10–1:15  Wrap-up (5 min)

### Optional / Take-Home Exercises
  - Cycles and asymmetry (exercise 3 in app)
  - Alpha and beta sensitivity (exercise 4 in app)
  - Substitutability k2 (exercise 5 in app)
  - Deeper experiment log analysis

### Battle Logistics

**Q-table encoding:**
  - Q1 = AdrenaLine (firm A), Q2 = BuzzFuel (firm B)
  - State s = s(pA, pB) for BOTH tables (always A's price first)
  - Q1[s,a] → A's action; Q2[s,a] → B's action
  - Demand is symmetric so roles are equivalent
  - If group submits Q2, flip states with flip_q_table_states to
    normalize to firm 1 perspective before matching

**Submission:**
  - Groups export Q-table from the app (need export button)
  - Submit via shared folder / upload link
  - Battle script matches all pairs, computes avg profit, ranks

**Open questions:**
  - Battle logistics: how do groups export/submit Q-tables?
  - Should battle be live in class or results shown next session?
  - Can groups submit multiple entries (different gammas)?
  - How to handle groups that don't converge?

## 11. Substitutability (k2) and Collusion

Ran 10 models per k2 at gamma=0.95 (100 total). k2 controls how easily
customers switch firms; higher k2 = more substitutable = bigger collusion premium.

| k2   | Premium% | Avg D1 | Avg D2 | Std  | Pattern              |
|------|----------|--------|--------|------|----------------------|
| 0.10 | 0.3%     | -0.40  | -0.19  | 0.90 | Noisy, no collusion  |
| 0.20 | 1.3%     | -0.01  | -0.22  | 1.11 | Noisy, no collusion  |
| 0.30 | 3.2%     | -0.15  | -0.39  | 1.07 | Noisy, no collusion  |
| 0.40 | 6.7%     | 0.45   | 0.49   | 0.33 | Collusion emerges    |
| 0.50 | 12.5%    | 0.62   | 0.62   | 0.14 | Stable collusion     |
| 0.60 | 22.5%    | 0.57   | 0.57   | 0.13 | Stable collusion     |
| 0.67 | 34.0%    | 0.57   | 0.57   | 0.13 | Stable collusion     |
| 0.75 | 56.3%    | 0.55   | 0.55   | 0.21 | Stable collusion     |
| 0.85 | 120.4%   | 2.49   | 2.28   | 0.92 | Super-collusive!     |
| 0.90 | 202.5%   | 3.66   | 3.49   | 1.12 | Super-collusive!     |

Key observations:
- **Low k2 (0.1-0.3)**: Collusion premium < 3%, algorithms can't find it.
  Delta is noisy, often negative. The price grid is narrow and concentrated
  near p_e ≈ p_c, so all prices are nearly equivalent. Converges fast
  (~180K steps) because the Q-table is easy to learn, but the outcome is
  essentially random within a narrow price band.
- **Medium k2 (0.4-0.75)**: Stable collusion at Delta ≈ 0.5-0.6. This is
  the sweet spot where the premium is large enough to matter and the
  algorithms can reliably coordinate. Variance is low.
- **High k2 (0.85-0.9)**: Delta goes ABOVE 1 — firms earn more than the
  theoretical joint-profit maximum! This happens because with near-perfect
  substitutes, the price grid extends far above p_c, and the algorithms
  find very high prices. But variance is large (asymmetric outcomes common).
  Premium is 120-200% so there's enormous value in coordination.
- **Non-monotonic in Delta, monotonic in absolute gains.** Delta ≈ 0.5 for
  k2=0.4 through 0.75, but absolute profit gain (Delta × premium) rises
  steadily with k2.
- **Convergence speed**: Low k2 converges in ~180K steps (tiny state space
  effectively). High k2 takes 1.5-2.5M steps.

### Why Delta > 1 at high k2

At k2=0.85-0.9 the algorithms converge to **long asymmetric cycles**
(length 2-10), NOT fixed points. Firms take turns charging very high and
very low prices. Example at k2=0.9 (p_e=10.9, p_c=46.5):

  (36.3, 82.1) → (82.1, 44.0) [2-cycle]

When firm 2 charges 82, firm 1 at 36 gets:
  q1 = 9 - 36 + 0.9×82 = 46.8, profit = (36-3)×46.8 = 1,544

This dwarfs profit_c = 189. In the cycle, each firm alternates between
being the "discount store" (low price, massive volume from cross-price
demand) and the "expensive brand" (high price, low volume). Average
profit across the cycle exceeds the symmetric collusive profit.

What's happening: in each period, the high-price firm has ZERO demand
(and zero profit) while the low-price firm captures all the demand.
They alternate who suffers. This is verified in Singh-Vives too.

When we require both firms to earn positive profit simultaneously,
symmetric collusion IS optimal — there is no improvement from asymmetry.
The super-collusion is a "take turns being the sacrifice" strategy that
yields high *average* profit but zero profit in half the periods.

This result is robust: it appears in both Calvano and Singh-Vives
demand specifications. It's driven by the fundamental property that
with high substitutability, the cross-price demand effect is so strong
that one firm can capture nearly all demand by undercutting.

Whether this is realistic is debatable — firms typically can't sustain
periods of zero profit voluntarily. But it's a genuine Nash equilibrium
of the repeated game (each firm's Q-values correctly reflect that the
"sacrifice" period is followed by the "harvest" period).

### Edge case issues with the Calvano demand model

The demand q_i = k1 - p_i + k2·p_j is an ad hoc linear specification
that does NOT properly nest the standard edge cases:

**k2 = 0 (independent goods):** p_e = p_c = (k1+c)/2 (monopoly).
  The price grid [2p_e-p_c, 2p_c-p_e] collapses to a single point.
  Model works but Delta is undefined (0/0) and the grid is useless.

**k2 → 1 (should be homogeneous Bertrand):** In this model, p_e =
  (k1+c)/(2-k2) → ∞ as k2→1. This is WRONG — homogeneous Bertrand
  should give p=c (the Bertrand paradox). The problem is that the
  own-price coefficient is normalised to 1 while cross-price is k2,
  so k2→1 doesn't make the goods homogeneous. Even at k2=0.99, a $1
  price cut only steals 0.99 units — it never captures the whole market.

**k2 ≥ 1 is nonsensical:** At symmetric pricing, q = k1 + (k2-1)p.
  If k2 ≥ 1, quantity INCREASES in price. The cross-price effect
  dominates the own-price effect, which is economically absurd.

### The microfounded alternative: Singh-Vives (1984)

The proper model starts from a representative consumer with utility:
  U = α(q1+q2) - ½(q1² + 2γ·q1·q2 + q2²)

where γ ∈ [0,1) is the substitutability parameter. This yields:
  q_i = (a - p_i + γ·p_j) / (1 - γ²)

where a = α(1-γ). The key ratio: cross-price / own-price = γ.

Edge cases:
  γ = 0 → independent monopolies (q_i = a - p_i, p* = (a+c)/2)
  γ → 1 → homogeneous goods (demands → ∞, p → c, Bertrand paradox)

This properly nests both extremes. Calvano's model is equivalent to
fixing the own-price coefficient at 1, which breaks the nesting.

### Calvano vs Singh-Vives: Q-learning equivalence

**Key result:** With a=k1 and γ=k2, Singh-Vives gives IDENTICAL Q-learning
dynamics to Calvano. Profits are scaled by 1/(1-k2²) but the greedy policy
(argmax) is invariant to positive scaling. Same price grid, same policies,
same Delta values.

To get genuinely different dynamics, use the "proper" SV parameterization:
  a = α(1-γ)  where α is fixed.

This changes the game because demand shrinks with γ:
  p_e → c as γ→1 (Bertrand paradox, correctly!)
  p_c = (α+c)/2 is CONSTANT for all γ
  Grid stays bounded (~[0, 12] at γ=0.95 with α=12)

With α=12, c=3:
| γ    | p_e  | p_c  | pi_e  | pi_c  | Premium |
|------|------|------|-------|-------|---------|
| 0.0  | 7.50 | 7.50 | 20.25 | 20.25 | 0%      |
| 0.3  | 6.71 | 7.50 | 15.09 | 15.58 | 3%      |
| 0.5  | 6.00 | 7.50 | 12.00 | 13.50 | 12%     |
| 0.67 | 5.23 | 7.50 | 9.05  | 12.13 | 34%     |
| 0.8  | 4.50 | 7.50 | 6.25  | 11.25 | 80%     |
| 0.9  | 3.82 | 7.50 | 3.52  | 10.66 | 203%    |
| 0.95 | 3.43 | 7.50 | 1.88  | 10.38 | 451%    |

### Singh-Vives Q-learning results (experiments_sv.csv)

Ran 10 models per γ at discount_factor=0.95, α=12, c=3.

| γ    | Prem% | p_e  | Avg D1 | Avg D2 | Std  | Cycle len |
|------|-------|------|--------|--------|------|-----------|
| 0.10 | 0%    | 7.26 | 0.13   | -0.39  | 0.64 | 10.3      |
| 0.30 | 3%    | 6.71 | -0.18  | -0.55  | 0.87 | 8.2       |
| 0.50 | 12%   | 6.00 | 0.57   | 0.57   | 0.13 | 1.0       |
| 0.60 | 22%   | 5.57 | 0.55   | 0.55   | 0.11 | 1.1       |
| 0.67 | 34%   | 5.23 | 0.51   | 0.51   | 0.17 | 1.0       |
| 0.80 | 80%   | 4.50 | 0.92   | 1.07   | 0.40 | 2.1       |
| 0.90 | 203%  | 3.82 | 3.45   | 3.83   | 1.09 | 3.2       |
| 0.95 | 451%  | 3.43 | 7.58   | 7.10   | 2.26 | 3.0       |

**Key comparison with Calvano:** Qualitatively identical patterns.
- Low γ: noisy, no collusion (same as Calvano low k2)
- Middle γ: stable Delta ≈ 0.5 symmetric fixed points (same)
- High γ: super-collusive asymmetric alternating cycles (same!)
  Delta > 1 persists even with bounded SV grid.

The "take turns earning zero" pattern confirmed in SV: in each period
of the cycle, the high-price firm has zero demand/profit, the low-price
firm captures massive demand. Not a Calvano artefact.

**Low γ cycle lengths:** At γ=0.1 (avg len 10.3) and γ=0.3 (avg 8.2),
the algorithms find long noisy cycles — not because of complex strategies,
but because the grid is very narrow (0.71 total range at γ=0.1) and all
prices give nearly identical profits. The "cycles" are random walks
on an essentially flat profit landscape.

For teaching: the Calvano model works well in the range k2 ∈ [0.3, 0.75]
where the interesting collusion dynamics happen. The edge cases are
pedagogically less important — students won't test k2=0 or k2=0.99.
Switching to Singh-Vives would add complexity without much teaching
benefit, since the Q-learning dynamics are similar in the interior.

## 12. Collusion Sustainability (Exercise 3 data)

Ran 15 models per gamma (0.95 and 0.8), testing stability, full collusion,
and all deviations with net gain calculations.

**Full collusion (p_c, p_c) is always abandoned:**
- gamma=0.95: 14/14 models return to partial cycle in 3.1 steps avg
- gamma=0.8: 14/14 models return to partial cycle in 2.8 steps avg
- Typical paths: (14.7,14.7) → (8-10, 8-10) → ... → (10.769,10.769)
- Algorithms learn "if both prices are very high, undercut aggressively"

**All deviations return to cycle (100%):**
- gamma=0.95: 196 deviations tested, all return, avg 3.0 steps punishment
- gamma=0.8: 196 deviations tested, all return, avg 2.3 steps punishment

**Self-enforcement:**
- gamma=0.95: almost all undercuts unprofitable (max net gain +3.79, rare)
  All overcuts unprofitable.
- gamma=0.8: ALL deviations unprofitable — fully self-enforcing.

## Key Economic Insights

1. **Algorithmic collusion emerges without communication** — Q-learning
   independently discovers supra-competitive pricing
2. **Patience is the key driver** — maps directly to folk theorem intuition
3. **Partial collusion, not full** — algorithms capture ~50% of the
   collusive premium at gamma=0.95, not 100%
4. **Punishment without design** — algorithms learn to punish deviations
   and forgive, without being programmed to do so
5. **Punishment depth limits collusion level** — the algorithms don't learn
   recursive punishments (punish failure to punish) needed for full collusion
6. **Gradual punishment, sudden forgiveness** — prices drift down slowly
   during punishment but snap back to collusion in one step
7. **Self-enforcing equilibria** — the converged prices pass the incentive
   compatibility test (no profitable deviation), except for rare fragile
   high-collusion outcomes
8. **Over-competition at low gamma** — myopic firms price below Nash,
   which is theoretically unexpected and practically interesting
9. **Bounded rationality sustains fragile collusion** — imprecise Q-values
   at off-equilibrium states can sustain prices that wouldn't survive
   exact best-response calculations
10. **Substitutability has a threshold effect** — collusion only emerges
    when products are substitutable enough (k2 ≥ 0.4) for the premium to
    be worth coordinating on. With near-perfect substitutes (k2 ≥ 0.85),
    algorithms achieve super-collusive outcomes (Delta > 1)
