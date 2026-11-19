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
