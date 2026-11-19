# Lesson 1: Kalman filter

What each simulation shows, the command, and the numbers you should see. All runs use the default seed, so the
numbers are exact; add `--lang tr` for Turkish text and figures, `--out <dir>` to choose where the files go.
The summary names (`std_filter`, `P_last`, ...) are the names the code uses.

## 01-fusion: fuse two measurements

A good thermometer reads 20 degrees with variance 1, a guess by eye says 30 with variance 9. Weighting each by one over
its variance gives one estimate that is better than both. Run `uv run fbh run 01-fusion`; expect `mean 21` and
`var 0.9` (the weights are 0.9 and 0.1, see the table in `out/01-fusion.csv`).

## 03-1d-loop: predict + update, a measurement gap

The loop by hand: start at 200 px with variance 4, speed 10, Q = 5, R = 1. In frame 2 the camera sees 216. Run
`uv run fbh run 03-1d-loop`; expect the prediction 210 (variance 9) to become `x_plus_frame_2 215.4` with
`P_plus_frame_2 0.9`, and, with no camera in frames 3 to 5, the variance to climb back to `P_last 15.9` at
`x_last 245.4`. Try `--q 0.5` or `--r 4` and watch the gain move.

## 05-derivative-vs-filter: why velocity belongs in the state

A target that does not move, seen by a detector that jitters by 3 px, 200 frames. Velocity from the difference of two
boxes is noisier than the boxes themselves; the filter's velocity is not. Run
`uv run fbh run 05-derivative-vs-filter`; expect `std_position 2.884`, `std_derivative 4.006` (theory,
sqrt(2) times sigma: `theory_derivative 4.243`) and `std_filter 0.171`, a ratio of about 20 (23 for this seed; try
`--seed 1`).

## 08-covariance: the covariance matrix P, before and after F

Errors drawn from P = [[2.5, 1], [1, 0.5]], then pushed through F = [[1, 1], [0, 1]]: the cloud shears, and its
covariance is F P F-transposed. Run `uv run fbh run 08-covariance`; expect `P_after_formula [[5, 1.5], [1.5, 0.5]]`,
with the 20000-sample estimate `P_after_empirical [[5.018, 1.501], [1.501, 0.499]]` right next to it.

## 12-velocity-learning: the filter learns a speed it never measures

The truth is 13 px/frame, the filter starts believing 10, and the camera only ever sees position. Run
`uv run fbh run 12-velocity-learning`; expect the speed estimate to reach `v_30 12.937` with `sigma_v_30 0.2`
(it started at `sigma_v_1 1.493` after the first frame), and `std_v_10_30 0.131`, the wobble of the estimate between
frames 10 and 30. The table in `out/12-velocity-learning.csv` is the one printed in the essay.

## 13-q-r-tradeoff: calm or agile

The same noise through a calm filter (q = 0.01) and an agile one (q = 1, change it with `--q`). Run
`uv run fbh run 13-q-r-tradeoff`; expect the calm filter's speed to wobble by `std_v_10_30_small 0.131` against
`std_v_10_30_big 0.442` for the agile one, and the gain on speed at frame 30 to be `K_v_30_small 0.08` against
`K_v_30_big 0.493`. On the right, the speed jumps from 10 to 14 at frame 36: the calm filter needs `settle_small 8`
frames to settle within 1 px/frame, the agile one touches that band after `reach_big 2` frames but jitters
(`calm_std_big 0.596` against `calm_std_small 0.112` in straight flight). `settle_big` depends on the seed; do not read
much into it.

## 14-nis: is the filter honest?

Three runs of the same filter (Q = 0.01, R = 4) over 400 frames, each scored by its normalised innovation squared (NIS),
which should average 1 and exceed the chi-square 95 % bound 3.84 in about 5 % of frames. Run `uv run fbh run 14-nis`;
expect `honest_mean 1.045` (`honest_above 0.06`), then an overconfident filter whose sensor really has R = 20:
`overconfident_mean 4.867`, above the bound in `overconfident_above 0.36` of the frames, and a filter whose Q is too
small for a speed jump at frame 200: `q_small_inside_mean 6.851` for the 20 frames after the jump and
`q_small_outside_mean 1.009` everywhere else. Other seeds scatter around these (seeds 0 to 19: honest 0.86 to 1.13,
overconfident 3.9 to 5.2); the sensor R is five times `--r`.
