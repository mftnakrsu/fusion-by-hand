# fusion-by-hand

Companion code for **Sensor Fusion from First Principles**, a course on [mefby.com](https://www.mefby.com/essays/fusion).
Lesson 1 builds a Kalman filter from scratch, with the numbers worked out by hand first. Lesson 2 takes it to a radar
that measures range and bearing, which no fixed matrix H can describe: the extended Kalman filter (EKF). This
repository holds both filters (`fbh/kf.py` and `fbh/ekf.py`, plain numpy), the seeded simulations behind every figure,
and a small command line tool, `fbh`, that runs them and tells you where you are in the course. The essays are also
available in Turkish ([lesson 1](https://www.mefby.com/essays/fusion-01-kalman-filter-tr),
[lesson 2](https://www.mefby.com/essays/fusion-02-extended-kalman-filter-tr)); `list`, `where` and `run` take
`--lang tr`.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer.

```sh
git clone https://github.com/mftnakrsu/fusion-by-hand
cd fusion-by-hand
uv sync
uv run fbh list
uv run fbh run 03-1d-loop
```

`fbh run` writes `out/03-1d-loop.png` (the figure), `out/03-1d-loop.csv` (the table behind it), prints the numbers
the essay quotes and points to the essay section and the next simulation.

Your progress is kept in `.fbh-progress.json` in the current directory, so run `fbh` from the same folder each time
(or point `FBH_PROGRESS=/some/where/progress.json` at one file to keep it elsewhere). `uv run fbh where` says how far
you are in the lesson you are working on (the lesson of the last simulation you ran for the first time) and what to
run next; when that lesson is done, it points to the next one. A progress file that cannot be read is ignored with a
warning on stderr, and the next `fbh run` writes a new one.

## Lesson 1: Kalman filter

Each id is one simulation, tied to a section of [the essay](https://www.mefby.com/essays/fusion-01-kalman-filter).

| id | what it shows | essay section |
|---|---|---|
| `01-fusion` | fuse two measurements | [1. Fusing two measurements](https://www.mefby.com/essays/fusion-01-kalman-filter#1-fusing-two-measurements) |
| `03-1d-loop` | predict + update, a measurement gap | [3. The one-dimensional loop](https://www.mefby.com/essays/fusion-01-kalman-filter#3-the-one-dimensional-loop) |
| `05-derivative-vs-filter` | why velocity belongs in the state | [5. Why velocity belongs in the state](https://www.mefby.com/essays/fusion-01-kalman-filter#5-why-velocity-belongs-in-the-state) |
| `08-covariance` | the covariance matrix P, before and after F | [8. The covariance matrix P](https://www.mefby.com/essays/fusion-01-kalman-filter#8-the-covariance-matrix-p) |
| `12-velocity-learning` | the filter learns a speed it never measures | [12. Python from scratch](https://www.mefby.com/essays/fusion-01-kalman-filter#12-python-from-scratch) |
| `13-q-r-tradeoff` | tuning Q and R: calm or agile | [13. Tuning Q and R](https://www.mefby.com/essays/fusion-01-kalman-filter#13-tuning-q-and-r) |
| `14-nis` | is the filter honest? the NIS test | [14. Is the filter honest? The NIS test](https://www.mefby.com/essays/fusion-01-kalman-filter#14-is-the-filter-honest-the-nis-test) |

What each simulation shows and which numbers to expect is in
[lesson-01-kalman-filter/README.md](lesson-01-kalman-filter/README.md).

## Lesson 2: extended Kalman filter

Each id is one simulation, tied to a section of [the essay](https://www.mefby.com/essays/fusion-02-extended-kalman-filter).

| id | what it shows | essay section |
|---|---|---|
| `ekf-02-tangent` | a belief through a curve, and its tangent | [2. The tangent line](https://www.mefby.com/essays/fusion-02-extended-kalman-filter#2-the-tangent-line) |
| `ekf-03-range-only` | one EKF update with the range alone | [3. One update by hand](https://www.mefby.com/essays/fusion-02-extended-kalman-filter#3-one-update-by-hand) |
| `ekf-05-radar-update` | one update with range and bearing | [5. One radar update by hand](https://www.mefby.com/essays/fusion-02-extended-kalman-filter#5-one-radar-update-by-hand) |
| `ekf-07-flyby` | tracking a fly-by with the EKF | [7. Python from scratch](https://www.mefby.com/essays/fusion-02-extended-kalman-filter#7-python-from-scratch) |
| `ekf-08-wrap` | the bearing wraps: with the fix and without | [8. The angle wraps](https://www.mefby.com/essays/fusion-02-extended-kalman-filter#8-the-angle-wraps) |
| `ekf-09-consistency` | where the tangent fails: four cases | [9. Where the tangent fails](https://www.mefby.com/essays/fusion-02-extended-kalman-filter#9-where-the-tangent-fails) |

What each simulation shows and which numbers to expect is in
[lesson-02-extended-kalman-filter/README.md](lesson-02-extended-kalman-filter/README.md).

## The commands

```
uv run fbh list                      the simulations, what you have run (✓), what is next (▶)
uv run fbh where                     how far you are and what to run next
uv run fbh run <id> [options]        figure, table and numbers of one simulation
uv run fbh export <dir>              the JSON lesson 1's browser figures are drawn from
uv run fbh export <dir> --lesson 2   the same for lesson 2
```

Options of `fbh run`: `--q` (process noise Q), `--r` (measurement noise R), `--gap A:B` (frames A to B have no
measurement), `--sigma-r` (the radar's range noise in metres, more than 0), `--sigma-theta` (its bearing noise in
radians, more than 0), `--seed`, `--lang en|tr`, `--out <dir>` (default `out`). Leave an option out and the simulation
uses the essay's own value. A simulation only reads the options it has a use for and says so on stderr when you pass
another. `--gap` is read by `13-q-r-tradeoff` (60 frames) and `14-nis` (400 frames); it has to fit in those frames and
leave measurements, and for `14-nis` some of frames 200 to 219 as well. Lesson 2 builds its R from the radar's two
sigmas, R = diag(sigma_r², sigma_theta²), so it reads neither `--r` nor `--gap`. `--sigma-r` is read by
`ekf-03-range-only`, `ekf-05-radar-update`, `ekf-07-flyby` and `ekf-08-wrap`. `--sigma-theta` is read by
`ekf-05-radar-update`, `ekf-07-flyby` and `ekf-08-wrap`. In `ekf-07-flyby` and `ekf-08-wrap`, `--q` is the tracker's
process noise intensity in m²/s³.

## Export format

`uv run fbh export <dir>` writes the five JSON files of lesson 1, `uv run fbh export <dir> --lesson 2` the three of
lesson 2. Each part below is everything a port of that lesson's browser figures needs.

### Lesson 1

The essay's browser figures run the filter again in JavaScript on `tracking.json` and compare it with `kf-cases.json`.
The model, with the state [position, velocity] and one frame per time step:

- F = [[1, 1], [0, 1]], H = [1, 0], R = r, Q = q * [[1/3, 1/2], [1/2, 1]].
- Frames are 1-based, but the JSON arrays are 0-based: the array index is frame - 1. `noise` is unit normal and the
  measurement of frame k is z = p[k - 1] + sqrt(r) * noise[k - 1]. `gap` and `maneuver` are frame numbers (1-based).
- Start from the frame-0 prior `x0`, `P0`. Every frame, frame 1 included, first predicts: x = F x, P = F P Fᵀ + Q.
- Then, unless the frame is in the gap, it updates: y = z - H x, S = H P Hᵀ + R, K = P Hᵀ / S, x = x + K y,
  P = (I - K H) P. That is the plain form, not Joseph, and P is not symmetrised afterwards.
- A gap `[a, b]` is inclusive: frames a to b are predict-only. There is no measurement there, so z, y, S, K and NIS do
  not exist, and x, P are the prediction.
- `kf-cases.json` exports only x, P and K, one entry per frame (index = frame - 1). `x` and `P` are after the update
  (the prediction in a gap frame); `K` is `null` in a gap frame. There are no z, y, S or NIS arrays: z comes from
  `tracking.json` as above, y and S you compute while filtering.
- Floats agree to about 1e-9 relative, not bit for bit: compare with a tolerance. P is symmetric only up to the last digits.

The files (compact JSON; `tracking.json` and `kf-cases.json` at full float precision, the others at 10 digits):

- `tracking.json`: `n` (60 frames), `gap` (`[20, 27]`), `maneuver` (36: the speed goes from 10 to 14 px/frame in that
  frame), `p` (true position), `v` (true speed), `noise` (unit normal, one per frame), `x0` (`[200, 7]`), `P0`
  (`[[16, 0], [0, 9]]`).
- `kf-cases.json`: `cases`, 24 objects for q in 0.001, 0.01, 1, 10, r in 1, 4, 25 and gap true, false, in that order
  (gap varying fastest). Each has `q`, `r`, `gap` (`true`: use the gap of `tracking.json`; `false`: no gap, same noise)
  and per frame `x` (`[[p, v], ...]`), `P` (`[[[a, b], [c, d]], ...]`) and `K` (`[[K_p, K_v], ...]`, `null` in gap frames).
- `derivative.json`: a target that does not move. `k` (frames 2 to 200), `truth_p` (250), `truth_v` (0), `z` (the
  measurements), `derivative` (velocity from the difference of two measurements), `filter` (the filter's velocity),
  `summary` (`sigma`, `std_position`, `theory_derivative`, `std_derivative`, `std_filter`, `ratio`).
- `covariance.json`: `n` (2000 samples), `F`, `before` and `after` (`[[position error, velocity error], ...]`, before
  and after F), `P_before` (the covariance the samples are drawn from), `P_after_formula` (F P Fᵀ).
- `nis.json`: `k` (frames 1 to 400), `bound` (3.84, the chi-square 95 % bound), `maneuver` (200), `window` (`[200, 219]`),
  `honest`, `overconfident`, `q_small` (one NIS value per frame each), `summary` (`n`, `r`, `r_sensor`, `maneuver`,
  `bound`, `honest_mean`, `overconfident_mean`, `q_small_mean`, `q_small_inside_mean`, `q_small_outside_mean`,
  `honest_above`, `overconfident_above`, `q_small_above`).

### Lesson 2

The radar sits at the origin and reports [range, bearing] of a target at (px, py): range = sqrt(px² + py²) in metres,
bearing = atan2(py, px) in radians, in [-pi, pi). The fly-by and the crossing are finished runs, so the browser only
draws them; the single updates of sections 2, 3, 5 and 9 are small enough to compute in the browser, and
`ekf-cases.json` holds the same calculations from Python to compare with. The tracker, with the state
[px, py, vx, vy] and one scan per second:

- F = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]] and
  Q = q * [[1/3, 0, 1/2, 0], [0, 1/3, 0, 1/2], [1/2, 0, 1, 0], [0, 1/2, 0, 1]]: lesson 1's model once per axis.
  R = diag(sigma_r², sigma_theta²).
- Scans are 1-based, but the JSON arrays are 0-based: the array index is scan - 1. The reports `z` are given, already
  noisy; their bearing lies in [-pi, pi).
- Start from the scan-0 prior `x0`, `P0`. Every scan, scan 1 included, first predicts: x = F x, P = F P Fᵀ + Q.
- Then it updates with that scan's report, linearised at the prediction: z_hat = [r, atan2(py, px)] with
  r = sqrt(px² + py²), H = [[px / r, py / r, 0, 0], [-py / r², px / r², 0, 0]], y = z - z_hat with the bearing part
  wrapped into [-pi, pi) (wrap(a) = (a + pi) mod 2 pi - pi, so 359 degrees off is 1 degree off), S = H P Hᵀ + R,
  K = P Hᵀ S⁻¹, x = x + K y, P = (I - K H) P (the plain form, not symmetrised), NIS = yᵀ S⁻¹ y.
- `wrap.json` runs the same tracker with the same q, sigma_r and sigma_theta as `flyby.json`. Its `without_wrap` run
  skips the wrap and uses the plain difference of the two bearings, the mistake of section 8.
- Floats are written to 10 significant digits, the reports `z` too, and a bearing rounded in its tenth digit moves a
  point 700 m away by about 1e-7 m. A port run on these reports therefore agrees with the stored results to about
  1e-7, not bit for bit: compare with a tolerance (1e-6 relative and absolute is safe).

The files (compact JSON, floats to 10 significant digits):

- `flyby.json`: `n` (60 scans), `dt` (1 s), `seed`, `q` (0.1), `sigma_r` (5 m), `sigma_theta` (0.02 rad), `bound`
  (5.99, the chi-square 95 % bound with two degrees of freedom), the truth `p` (`[[px, py], ...]`) and `v` (its
  constant velocity), the prior `x0`, `P0`, and per scan `raw` (the report on its own, as a point:
  r [cos(bearing), sin(bearing)]), `z` (the report), `z_hat` (the report the prediction expected), `xm`, `Pm` (the
  prediction), `x`, `P` (after the update), `y` (the innovation, bearing wrapped), `S`, `K` (4 × 2), `nis`, `error`
  (the distance from x to the truth), `claimed` (sqrt(P[0][0] + P[1][1]), the sigma the filter claims for that
  distance), `sigma_along` and `sigma_across` (the filter's sigma along the line of sight to its estimate and across
  it); `summary` (`q`, `sigma_r`, `sigma_theta`, `rms_raw`, `rms_filter`, `error_45`, `claimed_45`, `v_last`,
  `nis_mean`, `nis_above`; the rms values are over scans 10 to 60, after the start has settled).
- `wrap.json`: `n` (30 scans), `p`, `v`, `x0`, `P0`, `z` (the reports, the same for both runs), `jump_scan` (the first
  scan whose plain bearing difference is more than pi away from zero, `null` if there is none), `jump_scans` (all such
  scans), `raw` and `wrapped` (the bearing innovation of `jump_scan` without and with the wrap, `null` without a jump),
  the two runs `with_wrap` and `without_wrap`, each with per scan `z_hat`, `xm`, `x`, `P`, `y`, `nis` and `error`;
  `summary` (`jump_scan`, `z_theta`, `z_hat_theta`, `raw`, `wrapped`, `max_error_wrapped`, `max_error_plain`).
- `ekf-cases.json`: four lists of cases, each case its inputs and then its outputs.
  - `tangent` (section 2, 15 cases): the belief N(`x_hat`, `sigma`²) about px on a line that passes the radar at
    `offset`, pushed through range = sqrt(px² + offset²). `h` and `H` are the range and its slope at `x_hat`,
    `lin_std` = |H| sigma is the spread the tangent predicts (its mean is `h`), `true_mean` and `true_std` are what
    really comes out, by numerical integration over ±8 sigma.
  - `range` (section 3, 16 cases): one update with the range alone. The prediction `x_minus`, `P_minus`, the reading
    `z` with noise variance `R`, and `offset`; then h = sqrt(x_minus² + offset²) (`h`), H = x_minus / h (`H`),
    `y` = z - h, `S` = H² P_minus + R, `K` = P_minus H / S, `x_plus` = x_minus + K y, `P_plus` = (1 - K H) P_minus,
    and `exact_mean`, `exact_var`, the answer of Bayes' rule on a grid of ±8 sqrt(P_minus).
  - `update` (sections 5 and 9, 36 cases): one update of a position estimate. The prediction `x` = [px, py] has
    covariance `prior_sigma`² I, the radar reads `dr` metres and `dtheta` radians more than the prediction expects,
    z = z_hat + [dr, dtheta], with noise `sigma_r`, `sigma_theta`. The update is the tracker's on two states:
    `H` (2 × 2), `z_hat`, `z`, `y`, `S`, `K`, `x_plus`, `P_plus`, `nis`; then `var_along` = uᵀ P_plus u and
    `var_across` = tᵀ P_plus t, with u = x / |x| along the line of sight and t = [-u_y, u_x] across it, and
    `range_plus` = |x_plus|, to hold against the range the radar read. The first case is the hand calculation of
    section 5.
  - `exact` (36 cases): the same updates by Bayes' rule, without the tangent, and in the same order: `exact[i]` has the
    inputs of `update[i]`. With a_i, b_j = linspace(-`span`, `span`, `n`), the polar grid point r = z[0] + sigma_r a_i,
    bearing = z[1] + sigma_theta b_j, as p = r [cos(bearing), sin(bearing)], weighs
    exp(-|p - x|² / (2 prior_sigma²) - a_i² / 2 - b_j² / 2) r (0 where r ≤ 0); `mean` and `cov` are the weighted
    mean and covariance of p, `range_of_mean` = |mean|.

## Every simulated number and every simulation figure comes from these scripts

Each simulated number in the essay is either printed by `fbh run` or pinned by a test in `tests/`, and each simulation
figure is drawn from these scripts' output. The seeds are fixed, so you get the same numbers as the essay. A few small
tables in the essay are illustrative, hand-worked examples with made-up numbers; the essay says so where they appear.
If a simulated number in the text and a number you get ever disagree, one of them is a bug, so please open an issue.

```sh
uv run pytest -q
```

The code uses numpy and matplotlib only.

## License

MIT, see [LICENSE](LICENSE).
