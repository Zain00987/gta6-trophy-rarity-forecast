# Zain00987/gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Zain00987/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Zain00987/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast each GTA VI trophy's Exophase earn rate one week after launch — a Python program that reads a case's trophy reads and writes five quantiles per trophy.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9vs4v1v9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## The model

`main.py` (Python 3.13, standard library only, no network) forecasts each
trophy's Exophase earn rate at the case target on the logit scale:

- **Dilution + progression.** Exophase holds a snapshot, then jumps to a
  new one. On launch night the page sat at 638 players tracked, then
  refreshed to 3,868, and every rate but the one at 100% came down — newly
  tracked players have barely played, so a refresh dilutes almost every
  rate. The model collapses the history to distinct reads (refresh jumps),
  fits a power law for players-tracked growth, and regresses each trophy's
  logit-rate change on ln(players) across the jumps. Each trophy's slope
  (beta, shrunk toward a prior of −0.6) is its net dilution, combining new
  players pulling rates down with existing players progressing.
- **Flat history.** When no refresh has been seen yet, the model assumes
  the launch-night pattern is still pending: one refresh multiplying
  tracked players by about 6.06, so medians fall below the last rate for
  most trophies while the 100% trophy stays near 100%.
- **Honest spread.** The spread combines the baseline random-walk term
  with the scatter actually seen between refreshes, scaled by the number
  of refreshes expected before the target. It is never narrower than the
  baseline's — the baseline's band misses every refresh jump on the
  practice cases.

## Round 2

Three additions on top of round 1, all derived with Groq `gpt-oss-120b`
and then verified against the model code (two of its sketch formulas were
wrong — a split-normal density that did not integrate to 1, and a median
confused with the mode — and are implemented correctly here):

- **Count-consistent earners.** A rate is earners over players, and
  earners are integers that never decrease. Round 2 forecasts earners
  and players separately and takes the median rate as 100 * earners /
  players. Raw earners come from the beta drift; they are reconciled to
  the projected total trophies earned (ln T on ln P, shrunk toward a
  prior of 1 + mean beta, so the total coheres with the average drift)
  by water-filling with per-trophy bounds. A trophy already at 99% or
  more is exempt from the reconciliation — it stayed at 100% through a
  sixfold refresh on launch night, and an aggregate total that
  contradicts it is wrong, not a reason to cut it. When even the floors
  exceed the projected total, the total is infeasible and the drift
  stands unreconciled.
- **Refresh timing.** The answer is the page as of Exophase's last
  refresh before the target, so the effective horizon is the raw
  horizon minus half the refresh interval (from the distinct-read
  gaps), and the unknown lag adds a variance term
  (beta * observed growth rate * interval)^2 / 12, capped.
- **Asymmetric spread.** Refresh jumps skew down, so the spread is a
  two-piece normal on the logit scale, wider on the side the jump
  residuals lean toward, calibrated from their pooled skewness (shrunk
  toward a default of −0.15, the launch-night direction) and tilted
  mildly per trophy by its beta. The split normal is parameterised by
  its median, with the mode solved from the median explicitly.

## Practice validation

Run on the kit's practice cases `p00` and `p01` (flat histories, 638
players, 90-minute horizons):

- Both outputs validate: exactly one row per trophy (55), ids matching the
  case, five non-decreasing quantiles each in [0.01, 99.99], files under
  the 65,536-byte cap.
- Two runs write byte-identical output; each run takes well under a second
  (0.06 s for both cases together, limit 8 s per case).
- Compared with the starter baseline on the same cases, this model's
  medians sit below the last rate for every trophy (the 100% trophy stays
  at 99.99%), and its 90% bands are about 4.8x wider than the baseline's
  in logit units — wide enough to contain a launch-night-sized refresh
  jump, which the baseline's band cannot.

## Synthetic final-like test

The practice cases have flat histories, so they cannot tell round 1
from round 2. Both rounds were therefore also run on two synthetic
final-like cases (history from release to the deadline, target a week
after launch) whose ground truth comes from an independent
cohort/hazard process — players arriving on a power law, each earning
trophies with a per-trophy hazard per play hour, the page frozen
between 4-hourly refreshes — scored with the rules' exact pinball
formula. Mean pinball loss (lower is better):

| case | round 1 | round 2 |
|---|---|---|
| progression-dominated (rates rise) | 0.434378 | 0.373180 |
| dilution-dominated (rates fall) | 0.405697 | 0.298031 |

Round 2 beats round 1 in both regimes, with 90% coverage 1.000 in
both. The main fix during testing was the reconciliation: an early
version pinned the 100% trophy at 99.99% with a degenerate band (one
trophy costing 4.09 pinball on its own); the water-filling allocation
by headroom plus the saturated-trophy exemption removed that failure
mode.
