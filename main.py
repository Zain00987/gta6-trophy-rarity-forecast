"""gta6-trophy-rarity-forecast-01 entry, round 2: count-consistent earners,
refresh timing, asymmetric spread.

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

What round 1 did
----------------
Dilution + progression on the logit scale: a power law for the players
tracked, a per-trophy regression of the change in logit rate on the change
in ln(players) across the refresh jumps (beta, shrunk toward a prior of
-0.6), and a symmetric normal spread on the logit scale.  It works in rates
only and never looks at the counts underneath them.

What round 2 adds
-----------------
1.  COUNT-CONSISTENT earners.  A rate is earners over players, and earners
    are integers that never decrease.  Round 2 forecasts earners E_i and
    players P separately and takes the median rate as 100 * E_i / P.  The
    raw earners come from round 1's beta drift; they are then reconciled
    against the total trophies earned T, projected on its own from the
    history (ln T on ln P, shrunk toward a slope of 1.0), by iterative
    proportional fitting with per-trophy bounds: no trophy's earners may
    fall below what it had at the last read, none may exceed 99.99% of the
    players.  The reconciled earners are rounded to integers by largest
    remainder, preserving the projected total exactly.  A trophy already
    at 100% is no longer frozen there: its earners are bounded below by
    what it had, but its rate falls as new players arrive who have not
    earned it yet, which is what the launch-night refresh actually did.

2.  REFRESH TIMING.  The answer is the page as of Exophase's LAST refresh
    before the target, not the true rates at the target.  The last refresh
    is uniformly somewhere in the refresh interval I before the target, so
    the effective horizon is H - I/2 (I from the median gap between
    distinct reads), and all projections run to that horizon.  The unknown
    lag adds a variance term (beta * gdot * I)^2 / 12, where gdot is the
    players' log growth rate per hour.

3.  ASYMMETRIC spread.  Refresh jumps skew down: dilution drags most
    rates down, a few fast story trophies climb.  The spread is a
    two-piece (split) normal on the logit scale, wider on the side the
    residuals lean toward.  The split is calibrated from the pooled
    standardised third moment of the jump residuals, shrunk toward a
    default skew of -0.15 (the launch-night refresh moved every rate but
    one down); the side-sigma ratio is clipped to [0.5, 2.0] and tilted
    mildly per trophy by its beta.  The split normal is parameterised by
    its MEDIAN (the count-consistent point forecast), with the mode solved
    from the median explicitly -- in a split normal the mode is not the
    median, and the textbook density needs the factor 2/(s_lo + s_hi),
    not sqrt(2)/(s_lo + s_hi).  Both points matter for the quantile
    formula, and both are handled here rather than copied from a sketch.

Everything else is round 1's: the power-law players projection, the beta
regression with shrinkage, the spread floors and caps, the saturated
trophy's small symmetric spread.  The program is deterministic: no
randomness, no wall-clock reads, no network, standard library only.  Two
runs on the same input write byte-identical output.

Usage (local):  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader:        python3 -P -s /bot/main.py, with /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  This entry is one file, but the
# line keeps the behaviour identical to the kit's layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Release of the game, the anchor for the players power law.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Baseline random-walk spread, logit units per square-root day.
BASE_SIGMA = 0.5
# Prior slope of logit(rate) against ln(players): rates fall as the base grows.
BETA_PRIOR = -0.6
BETA_MIN, BETA_MAX = -1.5, 1.2
# Prior residual scatter of one refresh jump, logit units.
RESID_PRIOR_WITH_DATA = 0.6
RESID_PRIOR_FLAT = 0.6
# Launch-night refresh: 638 -> 3,868 players tracked.
FLAT_GROWTH = 3868.0 / 638.0
# Spread bounds, logit units.
SPREAD_MIN, SPREAD_MAX = 0.35, 2.5
SATURATED_SPREAD = 0.4
# Median shift bounds, logit units.
SHIFT_MIN, SHIFT_MAX = -3.5, 1.5
# Default refresh interval (hours) when the history shows no refresh yet.
FLAT_INTERVAL_H = 4.5
# Total-earners projection: slope of ln(T) on ln(P), shrunk toward a
# prior that coheres with the betas -- for small rates,
# dlnT/dlnP = 1 + beta, so the prior is 1 + mean(beta) (0.4 when the
# betas sit at their -0.6 prior, up to 2.0 when progression dominates).
# The ceiling is the 55-trophy cap: T cannot exceed 55 * P.
GAMMA_MIN, GAMMA_MAX = 0.0, 2.0
# Skew of the jump residuals: prior when no jumps are observed, and the
# weight that prior carries against the observed residuals.
SKEW_DEFAULT = -0.15
SKEW_PRIOR_WEIGHT = 10.0
SKEW_MAX = 0.8
# Side-sigma ratio from skew: R = clip(1 - KAPPA * gamma, 0.5, 2.0).
SKEW_KAPPA = 0.30
# Per-trophy tilt of the ratio by beta: R_i = R * (1 - TILT * beta_i).
BETA_TILT = 0.30


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


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9).

    Needed for the split-normal quantiles; the standard library has no
    inverse normal CDF.  p must lie strictly inside (0, 1).
    """
    p = min(1.0 - 1e-12, max(1e-12, p))
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3])
                                * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


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


def earners_of(read, index):
    """Earners of one trophy in one read, from its rate and players."""
    rate = read["rates"][index]
    if rate is None:
        return None
    return rate / 100.0 * float(read["players_tracked"])


def last_known_earners(history, index):
    """Most recent earners for one trophy, as an integer, or 0."""
    for read in reversed(history):
        value = earners_of(read, index)
        if value is not None:
            return int(round(value))
    return 0


def total_earners_of(read, n_trophies):
    """Total trophies earned in one read: the page's count, or the sum."""
    total = read.get("trophies_earned")
    if total is not None:
        return float(total)
    return sum(
        earners_of(read, i) or 0.0 for i in range(n_trophies)
    )


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
    gap = sorted(gaps)[len(gaps) // 2] if gaps else FLAT_INTERVAL_H

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

    Returns (betas, resid_sd, residuals).  Betas are shrunk toward the
    prior; the residual sd is the pooled scatter of refresh jumps around
    their own fitted slopes, shrunk toward a prior; residuals are the
    per-jump, per-trophy leftovers, used for the skew of the spread.
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
    residuals = []
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
            residuals.append(resid)
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
    return betas, resid_sd, residuals


def project_total_earners(reads, p_target, p_last, n_trophies, gamma_prior):
    """Project the total trophies earned at the target.

    Slope gamma of ln(T) on ln(P) over the distinct reads, shrunk toward
    gamma_prior (1 + mean beta: the total's growth coheres with the
    average trophy's drift, since T = P * mean rate).  Total earners
    grow with the players but slower when dilution dominates, faster
    when progression does.
    """
    t_last = total_earners_of(reads[-1], n_trophies)
    distinct = []
    seen = set()
    for read in reads:
        key = (read["players_tracked"],
               total_earners_of(read, n_trophies))
        if key not in seen:
            seen.add(key)
            distinct.append(read)
    gamma = gamma_prior
    if len(distinct) >= 3:
        xs, ys = [], []
        for read in distinct:
            p = float(read["players_tracked"])
            t = total_earners_of(read, n_trophies)
            if p > 0 and t > 0:
                xs.append(math.log(p))
                ys.append(math.log(t))
        n = len(xs)
        if n >= 3:
            mx = sum(xs) / n
            my = sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 1e-9:
                raw = sum((x - mx) * (y - my)
                          for x, y in zip(xs, ys)) / sxx
                gamma = (sxx * raw + 1.0 * gamma_prior) / (sxx + 1.0)
    gamma = min(GAMMA_MAX, max(GAMMA_MIN, gamma))
    ratio = max(1.0, p_target / p_last) if p_last > 0 else 1.0
    return t_last * (ratio ** gamma)


def reconcile_earners(raw, lo, total, p_target):
    """Reconcile raw earners to the projected total, as integers.

    Water-filling with per-trophy bounds: no trophy falls below its
    earners at the last read (earners never decrease), none exceeds
    99.99% of the players.  The difference from the raw total is
    allocated proportional to the raw earners, clipping to the bounds
    after each pass, so a trophy near its ceiling stops at its headroom
    instead of being pinned there.  Then largest-remainder rounding
    preserves the total exactly.  Monotonicity here is per trophy over
    time, not across trophies: trophy ids are Exophase list order, and a
    lower id does not mean more earners.
    """
    n = len(raw)
    hi = 0.9999 * p_target
    total_int = int(round(total))
    lo_sum = sum(lo)
    hi_sum = sum(int(math.floor(hi)) for _ in range(n))
    total_int = max(lo_sum, min(hi_sum, total_int))

    s_raw = sum(raw)
    if s_raw <= 0:
        base = [max(float(l), 1.0) for l in lo]
        s_base = sum(base)
        vals = [v * total_int / s_base for v in base]
    else:
        vals = [r for r in raw]
    # Water-filling: start from the raw earners and move toward the
    # total by allocating the remaining difference, clipping to the
    # bounds after each pass and reallocating whatever the capped
    # trophies could not take.  Growth is allocated by HEADROOM
    # (room to the 99.99% ceiling), not by size: a trophy already at
    # 97% of the players has almost no room left, and a proportional
    # share of a large total would pin it at the cap with a degenerate
    # band -- the logit scale cannot express any downside from 99.99%.
    # Shrinkage is allocated proportional to the raw earners: a
    # trophy's room to shrink is largest exactly where its earners
    # are largest, and cutting by that room would gut the trophies
    # that have earned the most.
    for _ in range(12):
        vals = [min(hi, max(float(lo[i]), v)) for i, v in enumerate(vals)]
        diff = total_int - sum(vals)
        if abs(diff) < 1e-6:
            break
        if diff > 0:
            free = [i for i, v in enumerate(vals) if v < hi - 1e-9]
            if not free:
                break
            weights = [hi - vals[i] for i in free]
            wsum = sum(weights)
            if wsum <= 0:
                share = diff / len(free)
                for i in free:
                    vals[i] += share
            else:
                for i, w in zip(free, weights):
                    vals[i] += diff * w / wsum
        else:
            free = [i for i, v in enumerate(vals)
                    if v > lo[i] + 1e-9]
            if not free:
                break
            base = sum(raw[i] for i in free)
            if base <= 0:
                share = diff / len(free)
                for i in free:
                    vals[i] += share
            else:
                for i in free:
                    vals[i] += diff * raw[i] / base

    floors = [int(math.floor(v)) for v in vals]
    # Flooring cannot break the lower bound (lo are integers) or the
    # upper bound, but check both anyway.
    floors = [min(int(math.floor(hi)), max(lo[i], f))
              for i, f in enumerate(floors)]
    remainder = total_int - sum(floors)
    order = sorted(range(n),
                   key=lambda i: (-(vals[i] - math.floor(vals[i])), i))
    k = 0
    while remainder > 0 and k < len(order):
        i = order[k]
        k += 1
        if floors[i] + 1 <= int(math.floor(hi)):
            floors[i] += 1
            remainder -= 1
    while remainder > 0:
        # Every trophy sits at its cap (cannot happen in practice):
        # put the rest on the largest trophy below the cap.
        placed = False
        for i in sorted(range(n), key=lambda j: -floors[j]):
            if floors[i] + 1 <= int(math.floor(hi)):
                floors[i] += 1
                remainder -= 1
                placed = True
                break
        if not placed:
            break
    return floors


def pooled_skew(residuals, resid_sd):
    """Skew of the jump residuals, shrunk toward the default.

    The launch-night refresh moved every rate but one down, so with no
    jumps observed the skew is mildly negative.  A wild estimate is
    ignored in favour of the default.
    """
    if len(residuals) >= 8 and resid_sd > 1e-9:
        observed = sum((r / resid_sd) ** 3 for r in residuals) / len(residuals)
        gamma = (len(residuals) * observed
                 + SKEW_PRIOR_WEIGHT * SKEW_DEFAULT) / (
                     len(residuals) + SKEW_PRIOR_WEIGHT)
    else:
        gamma = SKEW_DEFAULT
    if abs(gamma) > SKEW_MAX:
        gamma = SKEW_DEFAULT
    return max(-SKEW_MAX, min(SKEW_MAX, gamma))


def side_sigmas(spread, ratio):
    """Split-normal side sigmas from an overall sd and a ratio.

    The two-piece normal with mode m, left sd s_lo = R * k and right sd
    s_hi = k has variance k^2 * (R^2 + 1 - R - (2/pi)(1 - R)^2); solve
    for k so the variance equals spread^2.  At R = 1 this returns
    (spread, spread).
    """
    denom = math.sqrt(max(0.05, ratio * ratio + 1.0 - ratio
                          - (2.0 / math.pi) * (1.0 - ratio) ** 2))
    k = spread / denom
    return ratio * k, k


def split_normal_quantiles(median, s_lo, s_hi):
    """Quantiles of a two-piece normal, parameterised by its median.

    Density (2 / (s_lo + s_hi)) * phi((x - m) / s_side) either side of
    the mode m, so F(m) = w with w = s_lo / (s_lo + s_hi).  The median
    is NOT the mode: solve for m from the median, then invert the CDF.
    For tau <= w: x = m + s_lo * Phi^-1(tau / (2w)); above w:
    x = m + s_hi * Phi^-1(0.5 + (tau - w) / (2 (1 - w))).
    """
    w = s_lo / (s_lo + s_hi)
    if w >= 0.5:
        mode = median - s_lo * norm_ppf(1.0 / (4.0 * w))
    else:
        mode = median - s_hi * norm_ppf(
            0.5 + (0.5 - w) / (2.0 * (1.0 - w)))
    out = []
    for tau in LEVELS:
        if tau <= w:
            x = mode + s_lo * norm_ppf(tau / (2.0 * w))
        else:
            x = mode + s_hi * norm_ppf(
                0.5 + (tau - w) / (2.0 * (1.0 - w)))
        out.append(x)
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_h = case["horizon_minutes"] / 60.0

    reads = distinct_reads(history)

    # Refresh interval from the distinct-read gaps; the answer is the
    # page as of the last refresh before the target, so the effective
    # horizon is the raw horizon minus half an interval.
    _p, gap_h, _h = project_players(
        reads, case["target"], history[-1]["at"])
    interval = min(48.0, max(1.0, gap_h))
    horizon_eff_h = max(0.25 * horizon_h, horizon_h - interval / 2.0)
    t_last = parse_time(history[-1]["at"])
    t_eff = t_last + timedelta(hours=horizon_eff_h)
    p_target, _, _ = project_players(
        reads, t_eff.isoformat().replace("+00:00", "Z"), history[-1]["at"])
    p_last = float(reads[-1]["players_tracked"])
    growth = math.log(max(1.0, p_target / p_last))
    # Observed log growth rate of the players per hour, from the
    # distinct reads: the timing term prices the drift during the
    # unknown lag to the last refresh, so it needs the realised rate,
    # not the projected one (a flat history's projected jump is not a
    # rate at all).  No observed movement means no timing variance.
    gdot = 0.0
    if len(reads) >= 2:
        p_first = float(reads[0]["players_tracked"])
        span_h = (parse_time(reads[-1]["at"])
                  - parse_time(reads[0]["at"])).total_seconds() / 3600.0
        if p_first > 0 and span_h > 0:
            gdot = math.log(p_last / p_first) / span_h
            gdot = max(0.0, gdot)

    betas, resid_sd, residuals = estimate_drift(reads, n)
    skew = pooled_skew(residuals, resid_sd)
    ratio_pool = min(2.0, max(0.5, 1.0 - SKEW_KAPPA * skew))
    # The total-earners prior coheres with the betas: dlnT/dlnP = 1 +
    # mean(beta) for small rates.
    gamma_prior = min(2.0, max(0.0, 1.0 + sum(betas) / n))

    # Refreshes expected before the target: at least one is pending when
    # the history is flat, since a refresh is how the page ever moves.
    # Their scatter does not add up like independent coin flips: much of
    # a trophy's deviation from its fitted drift is a property of that
    # trophy, so the effective count grows only logarithmically.
    if len(reads) <= 1:
        n_refresh = max(1.0, horizon_eff_h / FLAT_INTERVAL_H)
    else:
        n_refresh = min(12.0, max(1.0, horizon_eff_h / interval))
    n_effective = 1.0 + 0.5 * math.log(n_refresh)

    base_spread = BASE_SIGMA * math.sqrt(max(horizon_days, 0.0))
    spread0 = math.sqrt(
        base_spread * base_spread + resid_sd * resid_sd * n_effective
    )
    spread0 = min(SPREAD_MAX, max(SPREAD_MIN, spread0))

    # Count-consistent median: raw earners from the beta drift, then
    # reconciled to the projected total trophies earned.
    raw_earners = []
    last_earners = []
    for i, trophy in enumerate(trophies):
        last_rate = last_known_rate(reads, i)
        if last_rate is None:
            last_rate = 0.5
        shift = betas[i] * growth
        shift = min(SHIFT_MAX, max(SHIFT_MIN, shift))
        centre = logit_of(last_rate) + shift
        raw_rate = 100.0 * sigmoid(centre)
        raw_earners.append(raw_rate / 100.0 * p_target)
        last_earners.append(last_known_earners(reads, i))
    t_target = project_total_earners(reads, p_target, p_last, n, gamma_prior)
    # A trophy already at 99% or more is exempt from the reconciliation:
    # its earners are pinned by its own drift (it stayed at 100%
    # through a sixfold refresh on launch night), and an aggregate
    # total that contradicts it -- a projected total smaller than this
    # one trophy's earners -- is simply wrong, not a reason to cut it.
    # The other trophies reconcile to what is left of the total; when
    # even that is below their floors, the total is infeasible and the
    # drift stands unreconciled.
    hi_all = 0.9999 * p_target
    saturated_idx = []
    for i, trophy in enumerate(trophies):
        last_rate = last_known_rate(reads, i)
        if last_rate is not None and last_rate >= 99.0:
            saturated_idx.append(i)
    earners = [0] * n
    for i in saturated_idx:
        earners[i] = int(round(min(hi_all, max(float(last_earners[i]),
                                                raw_earners[i]))))
    rest = [i for i in range(n) if i not in saturated_idx]
    if rest:
        remainder = t_target - sum(earners[i] for i in saturated_idx)
        floors_rest = sum(last_earners[i] for i in rest)
        if remainder < floors_rest:
            for i in rest:
                earners[i] = int(round(min(
                    hi_all, max(float(last_earners[i]), raw_earners[i]))))
        else:
            sub = reconcile_earners(
                [raw_earners[i] for i in rest],
                [last_earners[i] for i in rest],
                remainder, p_target)
            for k, i in enumerate(rest):
                earners[i] = sub[k]

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_known_rate(reads, i)
        if last_rate is None:
            last_rate = 0.5
        saturated = last_rate >= 99.0
        median_rate = 100.0 * earners[i] / p_target if p_target > 0 else 0.5
        centre = logit_of(median_rate)
        if saturated:
            sigma_lo = sigma_hi = SATURATED_SPREAD
        else:
            # Timing uncertainty: the last refresh lands anywhere in
            # [0, interval] before the target; its logit move has
            # variance (beta * gdot * interval)^2 / 12, capped so a
            # noisy rate cannot blow the spread up.
            timing_var = min(0.25, (betas[i] * gdot * interval) ** 2 / 12.0)
            spread = math.sqrt(spread0 * spread0 + timing_var)
            spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
            # Wider lower tail for dilution-prone trophies (beta < 0),
            # wider upper tail for progressing ones (beta > 0).
            ratio = min(2.0, max(0.5, ratio_pool
                                  * (1.0 - BETA_TILT * betas[i])))
            sigma_lo, sigma_hi = side_sigmas(spread, ratio)
        logits = split_normal_quantiles(centre, sigma_lo, sigma_hi)
        quantiles = []
        for x in logits:
            q = 100.0 * sigmoid(x)
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
