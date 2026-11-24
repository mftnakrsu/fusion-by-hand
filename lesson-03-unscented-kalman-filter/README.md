# Lesson 3: unscented Kalman filter

What each simulation shows, the command, and the numbers you should see. These are single steps worked by hand,
with no random draws, so the numbers are exact; add `--lang tr` for Turkish text and figures, `--out <dir>` to choose
where the files go. Each run prints its table (the sigma points, or the fields of a message) above the summary and
writes the same table to `out/<id>.csv`. The summary names (`z_hat`, `P_xz`, ...) are the names the code uses.
The sigma points are the unscaled set of `fbh/ukf.py`, with n + kappa = 3: kappa = 2 in one dimension, 1 in two.

## ukf-02-icd-decode: decode one LOS message of the example ICD

Before a filter can use a sensor, someone has to read the sensor's interface control document (ICD) and turn bytes
into angles. EOTS-X is a made-up electro-optical sensor with a made-up ICD; its line-of-sight (LOS) message is 32 bytes,
big-endian: an 8-byte header (sync, msg_id, version, payload length, sequence number) and a 24-byte payload (the
measurement time in µs, azimuth and elevation as int32 with an LSB of pi / 2³¹ rad, their rates as int16 with an LSB of
1e-4 rad/s, the LOS sigma in µrad, the sensor and a byte of validity bits). The CRC that follows on the wire is left
out. Run `uv run fbh run ukf-02-icd-decode`; the table gives each field's bytes, raw integer and value: msg_id
0x10 (LOS), length 24, seq 4711, azimuth 439875013 counts = 0.6435 rad, elevation 134932750 counts = 0.1974 rad,
el_rate -21 counts = -0.0021 rad/s, sigma_los 500 µrad, sensor MWIR, valid 0x0F. The summary holds the angles against
the lesson's geometry, a target 300 m east, 400 m north and 100 m up: `az_deg 36.87` against `atan_300_400_deg 36.87`
and `el_deg 11.31` against `atan_100_500_deg 11.31`, a quarter of a count apart (`az_off_lsb 0.234`,
`el_off_lsb -0.266`). Of the validity bits, `time_synced 1` and `gimbal_at_limit 0`.

## ukf-03-square: a belief through y = x²: tangent, three points, truth

A belief x ~ N(2, 1) pushed through y = x². The tangent at 2 (what the EKF does) says `ekf_mean 4` and `ekf_var 16`.
Three sigma points, 2 and 2 ± sqrt(3) with weights 2/3, 1/6, 1/6, each squared and averaged, say `ukf_mean 5` and
`ukf_var 18`, and the exact answer is `exact_mean 5` (mu² + sigma²) and `exact_var 18` (4 mu² sigma² + 2 sigma⁴).
Run `uv run fbh run ukf-03-square`; the table lists the three points, their weights and their squares.

## ukf-05-sigma-points: sigma points in one and two dimensions

In one dimension, N(300, 10²) gives 300 and 300 ± 17.32 (sqrt(3) × 10), weights 2/3 and 1/6. Run
`uv run fbh run ukf-05-sigma-points`; the three points give back `mean_1d 300` and `var_1d 100`, and their fourth
moment `m4_1d 30000` is a Gaussian's 3 sigma⁴ (`three_sigma4 30000`), which is why n + kappa = 3. In two dimensions,
P = 100 I around (300, 400) gives five points, the centre with weight 1/3 and (300 ± 17.32, 400), (300, 400 ± 17.32)
with 1/6 each. The tilted covariance lesson 2 ended with, `P_cholesky [[39.2, -14.4], [-14.4, 30.8]]` around
`mean_cholesky [300.8, 409.4]`, takes the columns of its Cholesky factor `L_cholesky [[6.261, 0], [-2.3, 5.051]]`;
along its eigen-axes it has `axis_var [20, 50]` in the directions `axis_dir [[0.6, 0.8], [-0.8, 0.6]]` (along the line
of sight and across it), so the points sit `axis_offset [7.746, 12.247]` m from the centre. Both sets give back the
same `mean_axes [300.8, 409.4]` and `P_axes [[39.2, -14.4], [-14.4, 30.8]]`: any square root of P will do.

## ukf-07-radar-update: one UKF update with the range alone

The update of lesson 2, section 3, without a Jacobian. The line passes 400 m north of the radar, the prediction says
x = 300 with variance 100, and the radar reads 506 m with sigma 2 m (`R 4`). The three points 282.68, 300 and 317.32
read 489.80, 500 and 510.58 m; their weighted mean is `z_hat 500.064`, their spread `spread 35.981`, so `S 39.981`.
The points' state and range move together by `P_xz 59.977`, the gain is `K 1.5`, the innovation `y 5.936`, and the
estimate becomes `x_plus 308.9049` with `P_plus 10.025` (printed with four decimals: it is 308.90, and three decimals,
308.905, would read as 308.91). Next to it the EKF of lesson 2: `ekf_x_plus 309`, `ekf_P_plus 10`; and Bayes' rule on
a fine grid: `exact_mean 308.914`, `exact_var 9.691`. This close in, the curve is straight enough for both. Run
`uv run fbh run ukf-07-radar-update`, and try `--sigma-r 10`.

## ukf-08-wide-belief: a wide belief: EKF and UKF part ways

The same line, but the prediction is 300 ± 100 m (`P_minus 10000`) and the radar reads 560 m. The EKF's tangent at 300
gives `ekf_x_plus 399.889` and claims `ekf_sigma 3.331`. The UKF's points, 126.8, 300 and 473.2, read 419.6, 500 and
619.6 m: `z_hat 506.538`, `S 3422.835`, `P_xz 5773.503`, `K 1.687`, and the estimate `x_plus 390.177` with
`sigma 16.17`. Bayes' rule says `exact_mean 391.827` with `exact_sigma 2.857`. The EKF is `ekf_off 8.062` m off and
claims a sigma that makes this `ekf_off_sigmas 2.42` sigmas: overconfident. The UKF is `ukf_off 1.651` m off,
`ukf_off_sigmas 0.102` of its sigma: honest, if cautious. Run `uv run fbh run ukf-08-wide-belief`.
