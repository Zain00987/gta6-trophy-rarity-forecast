"""gta6-trophy-rarity-forecast-01 entry: dilution + progression on the logit scale.

What the program does
---------------------
For every case it reads (one ``<id>.in`` file), it forecasts, for each of the
game's trophies, five quantiles of that trophy's earn rate on Exophase at the
case's target time.  The forecast is written to ``<id>.out`` as one JSON
object, ``{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}``.

Why the house baseline loses
----------------------------
The starter says every rate stays where the last read put it.  That is wrong
in a specific, predictable way on launch week: Exophase does not refresh its
counts continuously.  It holds a snapshot, then jumps to a new one.  On
launch night the page sat unchanged at 638 players tracked, then refreshed
to 3,868 players, and every rate but the one at 100% came down.  Newly
tracked players have barely played, so a refresh dilutes almost every rate.
The baseline's band is also far too narrow to contain a refresh jump, so on
the practice cases it scores about 1.51 and covers roughly nothing.

The model here
--------------
Work in earners and players, on the logit scale, where logit(p) =
ln(p / (100 - p)) with p in percent clamped to [0.01, 99.99]:

1.  Collapse the history to distinct reads.  Between refreshes the page is
    frozen, so consecutive identical reads carry no information; only the
    jumps do.

2.  Players.  Fit a power law, ln(players) against ln(hours since the
    game's release), over the distinct reads, and extrapolate it to the
    target.  The slope is clamped to [0, 1.5] and the projected multiple to
    [1x, 8x].  If the history is flat (no refresh yet seen), fall back to
    the launch-night pattern the rules state: one refresh that multiplies
    the tracked players by about 6.06, or a 0.7-power growth in time,
    whichever is larger.

3.  Per-trophy drift.  For each trophy, regress the change in its logit
    rate on the change in ln(players) across the refresh jumps, through the
    origin.  The slope beta is that trophy's net dilution: how much its
    logit rate moves per unit of player growth, combining dilution from new
    players with progression of the players already tracked.  Each beta is
    shrunk toward a prior of -0.6 (rates fall as the player base grows),
    except a trophy already at 99% or more, whose prior is 0.  Betas are
    clamped to [-1.5, 0.5].

4.  Median.  The median logit at the target is the last read's logit plus
    beta times ln(projected players / last players), with the total shift
    clamped to [-3.0, +1.5] logit units.

5.  Spread.  The spread is the baseline's random-walk term combined with
    the scatter actually seen between refreshes: the pooled residual
    standard deviation of the per-trophy regressions, times the square root
    of an effective refresh count that grows only logarithmically with the
    number of refreshes expected before the target (much of a trophy's
    deviation from its fitted drift is a property of that trophy, not a
    fresh coin flip per refresh).  It is never narrower than the
    baseline's spread, never wider than 2.5 logit units, and small (0.4)
    for a trophy already at 99% or more, which has nowhere to go but a
    hair down.

The program is deterministic: no randomness, no wall-clock reads, no
network, standard library only.  Two runs on the same input write
byte-identical output.

Usage (local):  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader:        python3 -P -s /bot/main.py, with /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  This entry is one file, but the
# line keeps the behaviour identical to the kit's layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release of the game, the anchor for the players power law.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Baseline random-walk spread, logit units per square-root day.
BASE_SIGMA = 0.5
# Prior slope of logit(rate) against ln(players): rates fall as the base grows.
BETA_PRIOR = -0.6
BETA_MIN, BETA_MAX = -1.5, 0.5
# Prior residual scatter of one refresh jump, logit units.
RESID_PRIOR_WITH_DATA = 0.6
RESID_PRIOR_FLAT = 0.6
# Launch-night refresh: 638 -> 3,868 players tracked.
FLAT_GROWTH = 3868.0 / 638.0
# Spread bounds, logit units.
SPREAD_MIN, SPREAD_MAX = 0.35, 2.5
SATURATED_SPREAD = 0.4
# Median shift bounds, logit units.
SHIFT_MIN, SHIFT_MAX = -3.0, 1.5


def clamp_rate(pct):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    """Hours from the game's release, floored so the log stays finite."""
    return max(0.5, (moment - RELEASE).total_seconds() / 3600.0)


def distinct_reads(history):
    """Collapse consecutive identical reads; only refresh jumps remain."""
    out = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if out and out[-1][0] == key:
            continue
        out.append((key, read))
    return [read for _, read in out]


def last_known_rate(history, index):
    """Most recent non-null rate for one trophy, or None if never seen."""
    for read in reversed(history):
        rate = read["rates"][index]
        if rate is not None:
            return rate
    return None


def project_players(reads, target, asof):
    """Project the players-tracked count at the target.

    Power-law fit of ln(players) on ln(hours since release) over the
    distinct reads; flat history falls back to the launch-night refresh
    pattern.  ``asof`` is the time of the last read in the case (a run of
    identical reads collapses to its first one, so the last read's time
    cannot be taken from ``reads``).  Returns (projected players, median
    gap between distinct reads in hours, horizon in hours).
    """
    last = reads[-1]
    p_last = float(last["players_tracked"])
    t_last = parse_time(asof)
    t_target = parse_time(target)
    horizon_h = max(0.0, (t_target - t_last).total_seconds() / 3600.0)

    gaps = []
    for a, b in zip(reads, reads[1:]):
        gaps.append(
            (parse_time(b["at"]) - parse_time(a["at"])).total_seconds() / 3600.0
        )
    gaps = [g for g in gaps if g > 0]
    gap = sorted(gaps)[len(gaps) // 2] if gaps else 4.5

    distinct_p = []
    seen = set()
    for read in reads:
        p = read["players_tracked"]
        if p not in seen:
            seen.add(p)
            distinct_p.append(read)

    ratio = 1.0
    if len(distinct_p) >= 2:
        xs, ys = [], []
        for read in distinct_p:
            xs.append(math.log(hours_since_release(parse_time(read["at"]))))
            ys.append(math.log(max(1.0, float(read["players_tracked"]))))
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 1e-9:
            alpha = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            alpha = min(1.5, max(0.0, alpha))
            h_last = hours_since_release(t_last)
            h_target = hours_since_release(t_target)
            ratio = (h_target / h_last) ** alpha
    if ratio <= 1.0:
        # Flat history: one launch-night-sized refresh is still pending.
        ratio = max(FLAT_GROWTH, (hours_since_release(t_target)
                                   / hours_since_release(t_last)) ** 0.7)
    ratio = min(8.0, max(1.0, ratio))
    return p_last * ratio, gap, horizon_h


def estimate_drift(reads, n_trophies):
    """Per-trophy beta of logit(rate) on ln(players), plus pooled scatter.

    Returns (betas, resid_sd).  Betas are shrunk toward the prior; the
    residual sd is the pooled scatter of refresh jumps around their own
    fitted slopes, shrunk toward a prior.
    """
    jumps = []  # (dlnP, [dlogit per trophy or None])
    for a, b in zip(reads, reads[1:]):
        pa = float(a["players_tracked"])
        pb = float(b["players_tracked"])
        if pa <= 0 or pb <= 0 or pa == pb:
            continue
        dlnp = math.log(pb / pa)
        dlogits = []
        for i in range(n_trophies):
            ra, rb = a["rates"][i], b["rates"][i]
            if ra is None or rb is None:
                dlogits.append(None)
            else:
                dlogits.append(logit_of(rb) - logit_of(ra))
        jumps.append((dlnp, dlogits))

    betas = []
    resid_sq_sum = 0.0
    resid_count = 0
    for i in range(n_trophies):
        sxx = 0.0
        sxy = 0.0
        for dlnp, dlogits in jumps:
            dy = dlogits[i]
            if dy is None:
                continue
            sxx += dlnp * dlnp
            sxy += dlnp * dy
        raw = sxy / sxx if sxx > 1e-9 else None
        # Shrink toward the prior; the prior weighs as much as 0.5 units
        # of summed squared log-player movement.
        prior = BETA_PRIOR
        if raw is None:
            beta = prior
        else:
            beta = (sxx * raw + 0.5 * prior) / (sxx + 0.5)
        betas.append(min(BETA_MAX, max(BETA_MIN, beta)))
        for dlnp, dlogits in jumps:
            dy = dlogits[i]
            if dy is None:
                continue
            resid = dy - betas[-1] * dlnp
            resid_sq_sum += resid * resid
            resid_count += 1

    if resid_count >= 4:
        observed = math.sqrt(resid_sq_sum / resid_count)
        resid_sd = math.sqrt(
            (resid_count * observed * observed
             + 4.0 * RESID_PRIOR_WITH_DATA ** 2) / (resid_count + 4.0)
        )
    elif jumps:
        resid_sd = RESID_PRIOR_WITH_DATA
    else:
        resid_sd = RESID_PRIOR_FLAT
    return betas, resid_sd


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0

    reads = distinct_reads(history)
    p_target, gap_h, horizon_h = project_players(
        reads, case["target"], history[-1]["at"]
    )
    p_last = float(reads[-1]["players_tracked"])
    growth = math.log(max(1.0, p_target / p_last))

    betas, resid_sd = estimate_drift(reads, n)

    # Refreshes expected before the target: at least one is pending when
    # the history is flat, since a refresh is how the page ever moves.
    # Their scatter does not add up like independent coin flips: much of
    # a trophy's deviation from its fitted drift is a property of that
    # trophy, so the effective count grows only logarithmically.
    if len(reads) <= 1:
        n_refresh = max(1.0, horizon_h / 4.5)
    else:
        interval = min(48.0, max(2.0, gap_h))
        n_refresh = min(12.0, max(1.0, horizon_h / interval))
    n_effective = 1.0 + 0.5 * math.log(n_refresh)

    base_spread = BASE_SIGMA * math.sqrt(max(horizon_days, 0.0))
    spread = math.sqrt(
        base_spread * base_spread + resid_sd * resid_sd * n_effective
    )
    spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_known_rate(reads, i)
        if last_rate is None:
            last_rate = 0.5
        saturated = last_rate >= 99.0
        beta = 0.0 if saturated else betas[i]
        shift = beta * growth
        shift = min(SHIFT_MAX, max(SHIFT_MIN, shift))
        centre = logit_of(last_rate) + shift
        sigma = SATURATED_SPREAD if saturated else spread
        quantiles = []
        for z in Z:
            q = 100.0 * sigmoid(centre + z * sigma)
            quantiles.append(round4(min(99.99, max(0.01, q))))
        # Rounding can only break monotonicity by a hair; restore it.
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
