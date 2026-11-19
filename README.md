# fusion-by-hand

Companion code for **Sensor Fusion from First Principles**, a course on [mefby.com](https://www.mefby.com/essays/fusion).
Lesson 1 builds a Kalman filter from scratch, with the numbers worked out by hand first. This repository holds the
filter (about 40 lines of numpy), the seeded simulations behind every figure, and a small command line tool, `fbh`,
that runs them and tells you where you are in the lesson. The essay is also available [in Turkish](https://www.mefby.com/essays/fusion-01-kalman-filter-tr); `list`, `where` and `run` take `--lang tr`.

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
you are and what to run next. A progress file that cannot be read is ignored with a warning on stderr, and the next
`fbh run` writes a new one.

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

Lesson 2 (EKF) is being written. What each simulation shows and which numbers to expect is in
[lesson-01-kalman-filter/README.md](lesson-01-kalman-filter/README.md).

## The commands

```
uv run fbh list                    the simulations, what you have run (✓), what is next (▶)
uv run fbh where                   how far you are and what to run next
uv run fbh run <id> [options]      figure, table and numbers of one simulation
uv run fbh export <dir>            the JSON the essay's browser figures are drawn from
```

Options of `fbh run`: `--q` (process noise Q), `--r` (measurement noise R), `--gap A:B` (frames A to B have no
measurement), `--seed`, `--lang en|tr`, `--out <dir>` (default `out`). Leave an option out and the simulation uses
the essay's own value. A simulation only reads the options it has a use for and says so on stderr when you pass another.
`--gap` is read by `13-q-r-tradeoff` (60 frames) and `14-nis` (400 frames); it has to fit in those frames and leave
measurements, and for `14-nis` some of frames 200 to 219 as well.

## Export format

`uv run fbh export <dir>` writes five JSON files. The essay's browser figures run the filter again in JavaScript on
`tracking.json` and compare it with `kf-cases.json`; this section is everything such a port needs.

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
