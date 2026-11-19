# Lesson 2: extended Kalman filter

What each simulation shows, the command, and the numbers you should see. All runs use the default seed, so the
numbers are exact; add `--lang tr` for Turkish text and figures, `--out <dir>` to choose where the files go.
The summary names (`rms_filter`, `P_plus`, ...) are the names the code uses. The radar sits at the origin and reports
range in metres and bearing in radians, with noise sigma 5 m and 0.02 rad unless you change them with `--sigma-r` and
`--sigma-theta`; one scan takes one second.

## ekf-02-tangent: a belief through a curve, and its tangent

A target on a line that passes the radar 400 m away is at range r = sqrt(px² + 400²): a curve, not a matrix. At
px = 300 the curve is at 500 m and climbs 0.6 m per metre, and the EKF puts that tangent in its place. Run
`uv run fbh run ekf-02-tangent`; expect `h 500` and `H 0.6`, and for a belief of 300 ± 10 m a range of
`true_mean 500.064` with `true_std 5.998`, next to the tangent's 500 and `lin_std 6`: here the tangent is all you
need. At px = 0 the tangent is flat (`flat_lin_std 0`), yet a belief of 0 ± 100 m comes out at `flat_true_mean 411.987`
with `flat_true_std 16.325`. The figure takes the second row of the table in `out/ekf-02-tangent.csv`, 300 ± 100 m:
the range really comes out at 506.5 m on average, not 500, and never below 400 m, where the tangent's bell still puts
some of it.

## ekf-03-range-only: one EKF update with the range alone

The prediction says px = 300 with variance 100; the radar, reading range only here, sees 506 m with sigma 2 m
(variance 4). By hand: the curve expects 500, so y = 6, S = 0.6² × 100 + 4 = 40 and K = 0.6 × 100 / 40 = 1.5. Run
`uv run fbh run ekf-03-range-only`; expect `y 6`, `S 40`, `K 1.5`, `x_plus 309` and `P_plus 10`, against
`exact_mean 308.914` and `exact_var 9.691` from Bayes' rule on a fine grid: this close to px = 300 the curve is
straight enough for the tangent. On its own the reading would put the target at sqrt(506² - 400²) = 309.9. Try
`--sigma-r 10` and watch the gain fall.

## ekf-05-radar-update: one update with range and bearing

The prediction is (300, 400) with variance 100 in every direction: range 500 m at bearing 0.9273 rad. The radar reads
10 m farther and 0.02 rad to the left, which at 500 m is 10 m across the line of sight. Run
`uv run fbh run ekf-05-radar-update`; expect `H [[0.6, 0.8], [-0.0016, 0.0012]]` (the first row along the line of
sight, the second across it divided by the range), `S [[125, 0], [0, 0.0008]]`, `K [[0.48, -200], [0.64, 150]]`,
`x_plus [300.8, 409.4]` and `P_plus [[39.2, -14.4], [-14.4, 30.8]]`, with `nis 1.3`. Along the line of sight the
variance falls from 100 to `var_along 20`, across it only to `var_across 50`: the radar's 5 m in range beats its
0.02 rad × 500 m = 10 m across. `--sigma-r` and `--sigma-theta` change the radar; the prediction and the reading stay.

## ekf-07-flyby: tracking a fly-by with the EKF

A UAV flies straight past the radar at 20 m/s, 400 m north of it, for 60 scans, and the EKF of section 7 follows it
from a start about one sigma off. Run `uv run fbh run ekf-07-flyby`; from scan 10 on, the reports on their
own miss the truth by `rms_raw 11.091` m and the filter by `rms_filter 5.039`. At scan 45, the point of the hand
calculation, the estimate is `error_45 4.762` m off and claims `claimed_45 5.264`; the speed ends at
`v_last [20.266, -0.163]`. The NIS averages `nis_mean 1.81` (an honest filter gives 2) and is above the 95 % bound
5.99 in `nis_above 0.05` of the scans. The default seed is 4, a typical draw: for seeds 0 to 19 the filter always
beats the reports, with rms 3.1 to 7.9 m against 9.7 to 13.5 m.

## ekf-08-wrap: the bearing wraps

A UAV crosses the line due west of the radar, where the bearing jumps from +pi to -pi. At scan 10 the target is
still 2.5 m north of that line; its reading, noise included, is already south of it, and the prediction north. Run
`uv run fbh run ekf-08-wrap`; expect `jump_scan 10`, `z_theta -3.1279` against `z_hat_theta 3.1266`, a plain
difference of `raw -6.2545` rad that is really `wrapped 0.0287` rad (1.65 degrees). With the wrap the estimate is never
more than `max_error_wrapped 12.622` m off; without it, it flies away, up to `max_error_plain 1159.482` m. Seeds 13
and 19 never put the reading and the prediction on opposite sides of the cut, so `jump_scan` is `none` there and
both runs are the same (try `--seed 13`); for every other seed from 0 to 19 the run without the wrap is at least 40
times worse.

## ekf-09-consistency: where the tangent fails

Four cases of 300 targets each, every target obeying the filter's own motion model, so whatever goes wrong is the
tangent's doing. An honest filter has NIS 2 on average (two degrees of freedom) and an rms error equal to the sigma it
claims (ratio 1). What decides it is the bend of the range circle across the first update's cross-range sigma,
sigma_c² / (2 r), against the range noise. Run `uv run fbh run ekf-09-consistency` (it takes a few seconds). With a
30 m prior the bend is `baseline_bend 0.639` m and the filter is honest (`baseline_nis_early 2.025`,
`baseline_nis_late 2.02`, `baseline_ratio 1.003`). With a 300 m prior it is `vague_prior_bend 63.87` m, far beyond the
5 m range noise: the first updates go wrong (`vague_prior_nis_early 98.06`) and the filter never quite recovers
(`vague_prior_nis_late 6.945`, `vague_prior_ratio 3.722`). Starting from the first report instead brings the bend down
to `first_report_bend 0.141` m and the filter back to honest (`first_report_ratio 1.022`). Started from a report with
a sharp range and a coarse bearing (0.5 m and 0.1 rad), the circle bends by `sharp_range_bend 3.523` m, seven times
its range noise, and the filter ends up overconfident (`sharp_range_nis_late 3.377`, `sharp_range_ratio 2.799`).
Other seeds (`--seed 1`) tell the same story with other decimals.
