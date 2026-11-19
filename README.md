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

`fbh run` writes `out/03-1d-loop.png` (the figure), `out/03-1d-loop.csv` (the table behind it) and prints the numbers
the essay quotes. Your progress is kept in `.fbh-progress.json`; `uv run fbh where` says what to run next.

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
fbh list                          the simulations, what you have run (✓), what is next (▶)
fbh where                         how far you are and what to run next
fbh run <id> [options]            figure, table and numbers of one simulation
fbh export <dir>                  the JSON the essay's browser figures are drawn from
```

Options of `fbh run`: `--q` (process noise Q), `--r` (measurement noise R), `--gap A:B` (frames A to B have no
measurement; `none` for no gap), `--seed`, `--lang en|tr`, `--out <dir>` (default `out`). Leave an option out and the
simulation uses the essay's own value. A simulation only reads the options it has a use for and says so on stderr when
you pass another.

## Every number in the essay comes from these scripts

Each number in the essay is either printed by `fbh run` or pinned by a test in `tests/`. The seeds are fixed, so you get
the same numbers as the essay. If a number in the text and a number you get ever disagree, one of
them is a bug, so please open an issue.

```sh
uv run pytest -q
```

The code uses numpy and matplotlib only.

## License

MIT, see [LICENSE](LICENSE).
