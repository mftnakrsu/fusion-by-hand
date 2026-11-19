import csv
import json
import math
import re
from pathlib import Path

import pytest

from fbh import registry, sims, sims2
from fbh.cli import main

ROOT = Path(__file__).resolve().parent.parent
ESSAY = "https://www.mefby.com/essays/fusion-01-kalman-filter"
ESSAY2 = "https://www.mefby.com/essays/fusion-02-extended-kalman-filter"
LESSON1 = ["01-fusion", "03-1d-loop", "05-derivative-vs-filter", "08-covariance", "12-velocity-learning",
           "13-q-r-tradeoff", "14-nis"]
LESSON2 = ["ekf-02-tangent", "ekf-03-range-only", "ekf-05-radar-update", "ekf-07-flyby", "ekf-08-wrap",
           "ekf-09-consistency"]


@pytest.fixture
def progress(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    monkeypatch.setenv("FBH_PROGRESS", str(path))
    return path


def run(capsys, *argv):
    code = main(list(argv))
    assert code in (0, None)
    return capsys.readouterr().out


def have_run(progress, ids):
    """A progress file as if `ids` had been run in this order (without spending the time to run them)."""
    progress.write_text(json.dumps({"run": list(ids)}) + "\n")


# --- the brief's three commands ----------------------------------------------------------

def test_list_run_where_follow_the_brief(progress, tmp_path, capsys):
    out = run(capsys, "list")
    lines = out.splitlines()
    assert lines[0] == "LESSON 1 · Kalman filter" and lines[8] == "LESSON 2 · EKF"
    assert len(lines) == 1 + 7 + 1 + 6
    assert lines[1] == "  ▶ 01-fusion                fuse two measurements"
    assert lines[9] == "    ekf-02-tangent           a belief through a curve, and its tangent"
    assert lines[-1] == "    ekf-09-consistency       where the tangent fails: four cases"
    assert "✓" not in out

    run(capsys, "run", "01-fusion", "--out", str(tmp_path / "o"))
    run(capsys, "run", "03-1d-loop", "--out", str(tmp_path / "o"))
    assert (tmp_path / "o" / "03-1d-loop.png").stat().st_size > 5000
    assert (tmp_path / "o" / "03-1d-loop.csv").exists()

    lines = run(capsys, "list").splitlines()
    assert lines[1] == "  ✓ 01-fusion                fuse two measurements"
    assert lines[2] == "  ✓ 03-1d-loop               predict + update, a measurement gap"
    assert lines[3] == "  ▶ 05-derivative-vs-filter  why velocity belongs in the state"
    assert lines[4].startswith("    08-covariance  ")

    assert run(capsys, "where") == ("Lesson 1 · 2 of 7 simulations run. "
                                    "Next: 05-derivative-vs-filter (why velocity belongs in the state).\n")


def test_where_from_nothing_to_everything(progress, tmp_path, capsys):
    assert run(capsys, "where") == "Lesson 1 · 0 of 7 simulations run. Next: 01-fusion (fuse two measurements).\n"
    for sid in LESSON1:
        run(capsys, "run", sid, "--out", str(tmp_path))
    assert run(capsys, "where") == ("Lesson 1 · 7 of 7 simulations run. Lesson 1 is done. "
                                    "Next: ekf-02-tangent (a belief through a curve, and its tangent).\n")
    assert run(capsys, "where", "--lang", "tr") == ("Ders 1 · çalıştırılan simülasyon: 7 / 7. Ders 1 bitti. "
                                                    "Sırada: ekf-02-tangent (bir inanç eğriden geçiyor, yanında teğeti).\n")
    assert "  ▶ ekf-02-tangent  " in run(capsys, "list")
    for sid in LESSON2:
        run(capsys, "run", sid, "--out", str(tmp_path))
    assert run(capsys, "where") == "Lesson 2 · 6 of 6 simulations run. Lesson 2 is done.\n"
    assert run(capsys, "where", "--lang", "tr") == "Ders 2 · çalıştırılan simülasyon: 6 / 6. Ders 2 bitti.\n"
    assert "▶" not in run(capsys, "list") and run(capsys, "list").count("✓") == 13


def test_where_and_next_follow_the_lesson_you_are_in(progress, tmp_path, capsys):
    """Lesson 2 without lesson 1: its own next steps, and once it is done, nothing points back."""
    out = run(capsys, "run", "ekf-03-range-only", "--out", str(tmp_path))
    assert out.splitlines()[-1] == "Next: ekf-02-tangent (a belief through a curve, and its tangent)"
    assert run(capsys, "where") == ("Lesson 2 · 1 of 6 simulations run. "
                                    "Next: ekf-02-tangent (a belief through a curve, and its tangent).\n")
    assert "  ▶ ekf-02-tangent  " in run(capsys, "list")
    out = run(capsys, "run", "03-1d-loop", "--out", str(tmp_path))                 # over to lesson 1
    assert out.splitlines()[-1] == "Next: 01-fusion (fuse two measurements)"
    assert run(capsys, "where") == "Lesson 1 · 1 of 7 simulations run. Next: 01-fusion (fuse two measurements).\n"
    have_run(progress, ["ekf-03-range-only", "03-1d-loop"] + [sid for sid in LESSON2[2:]])
    out = run(capsys, "run", "ekf-02-tangent", "--out", str(tmp_path))
    assert out.splitlines()[-1] == "Lesson 2 is done."
    assert run(capsys, "where") == "Lesson 2 · 6 of 6 simulations run. Lesson 2 is done.\n"
    assert "▶" not in run(capsys, "list")
    out = run(capsys, "run", "03-1d-loop", "--out", str(tmp_path))                 # a rerun moves nothing
    assert out.splitlines()[-1] == "Next: 01-fusion (fuse two measurements)"
    assert run(capsys, "where") == "Lesson 2 · 6 of 6 simulations run. Lesson 2 is done.\n"
    have_run(progress, LESSON2 + LESSON1[1:])
    out = run(capsys, "run", "01-fusion", "--out", str(tmp_path))
    assert out.splitlines()[-1] == "Lesson 1 is done."                             # and lesson 2 was done already
    assert run(capsys, "where") == "Lesson 1 · 7 of 7 simulations run. Lesson 1 is done.\n"


def test_a_lesson_still_being_written_is_announced(progress, tmp_path, capsys, monkeypatch):
    soon = registry.Lesson(3, ("UKF", "UKF"), ("https://example.com/3", "https://example.com/3-tr"), ())
    monkeypatch.setattr(registry, "LESSONS", registry.LESSONS + (soon,))
    assert run(capsys, "list").splitlines()[-1] == "LESSON 3 · UKF (writing)"
    assert run(capsys, "list", "--lang", "tr").splitlines()[-1] == "DERS 3 · UKF (yazılıyor)"
    have_run(progress, LESSON2[1:])
    out = run(capsys, "run", "ekf-02-tangent", "--out", str(tmp_path))
    assert out.splitlines()[-1] == "Lesson 2 is done. Lesson 3 (UKF) is being written."
    assert run(capsys, "where", "--lang", "tr") == ("Ders 2 · çalıştırılan simülasyon: 6 / 6. Ders 2 bitti. "
                                                    "Ders 3 (UKF) yazılıyor.\n")


def test_progress_file_is_json_unique_and_survives_reruns(progress, tmp_path, capsys):
    for sid in ("03-1d-loop", "03-1d-loop", "01-fusion"):
        run(capsys, "run", sid, "--out", str(tmp_path))
    assert json.loads(progress.read_text())["run"] == ["03-1d-loop", "01-fusion"]


def test_progress_defaults_to_a_file_in_the_working_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FBH_PROGRESS", raising=False)
    monkeypatch.chdir(tmp_path)
    run(capsys, "run", "01-fusion", "--out", "o")
    assert (tmp_path / ".fbh-progress.json").exists()


def test_progress_creates_its_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FBH_PROGRESS", str(tmp_path / "deep" / "er" / "p.json"))
    run(capsys, "run", "01-fusion", "--out", str(tmp_path))
    assert json.loads((tmp_path / "deep" / "er" / "p.json").read_text()) == {"run": ["01-fusion"]}


def test_a_broken_progress_file_warns_on_stderr_and_counts_as_nothing_run(progress, tmp_path, capsys):
    progress.write_text("{ not json")
    main(["where"])
    cap = capsys.readouterr()
    assert "0 of 7" in cap.out
    assert cap.err.count("\n") == 1 and "warning" in cap.err and str(progress) in cap.err      # one line, names the file
    main(["list"])
    assert "warning" in capsys.readouterr().err
    assert progress.read_text() == "{ not json"                                  # reading never overwrites it
    main(["run", "01-fusion", "--out", str(tmp_path)])                           # a run rewrites it, after saying so
    assert capsys.readouterr().err.count("warning") == 1
    assert json.loads(progress.read_text()) == {"run": ["01-fusion"]}
    main(["where"])
    assert capsys.readouterr().err == ""
    for bad in ('{"run": "01-fusion"}', "[]", ""):                                # valid JSON of the wrong shape, empty file
        progress.write_text(bad)
        main(["where"])
        assert "warning" in capsys.readouterr().err, bad


def test_a_missing_progress_file_is_not_a_warning(progress, capsys):
    main(["where"])
    assert capsys.readouterr().err == ""


def test_an_unwritable_out_is_a_readable_usage_error_and_not_recorded(progress, tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    for argv in (["run", "01-fusion", "--out", str(blocker / "sub")], ["export", str(blocker / "sub")]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2
        err = capsys.readouterr().err
        assert "Traceback" not in err and str(blocker) in err and "error" in err
    assert not progress.exists()


# --- what run writes and prints ----------------------------------------------------------

def test_run_writes_a_csv_of_the_table_and_a_png(progress, tmp_path, capsys):
    out = run(capsys, "run", "03-1d-loop", "--out", str(tmp_path))
    rows = list(csv.DictReader((tmp_path / "03-1d-loop.csv").read_text().splitlines()))
    assert [r["k"] for r in rows] == ["1", "2", "3", "4", "5"]
    assert list(rows[0]) == ["k", "x_minus", "P_minus", "z", "K", "x_plus", "P_plus"]
    assert rows[0]["x_minus"] == "" and rows[1]["K"] == "0.9" and float(rows[1]["x_plus"]) == pytest.approx(215.4)
    assert rows[1]["P_plus"] == "0.9" and rows[0]["x_plus"] == "200"          # 0.8999999999999998 is not shown
    assert "03-1d-loop.png" in out and "03-1d-loop.csv" in out


def test_run_prints_rounded_summary_numbers(progress, tmp_path, capsys):
    out = run(capsys, "run", "01-fusion", "--out", str(tmp_path))
    assert re.search(r"^\s+mean\s+21$", out, re.M) and re.search(r"^\s+var\s+0\.9$", out, re.M)
    for sid in sims.SIMS:
        text = run(capsys, "run", sid, "--out", str(tmp_path))
        summary = text.split("\nwrote ")[0]
        if sid == "ekf-08-wrap":                                             # its bearings print 4 decimals, see below
            summary = "\n".join(line for line in summary.splitlines() if not re.match(r"\s+(z_|raw|wrapped)", line))
        for number in re.findall(r"\d+\.\d+", summary):                    # 3 decimals at most, two significant
            whole, decimals = number.split(".")                              # digits for a value under 0.01
            assert len(decimals) <= 3 or (whole == "0" and decimals.startswith("00")
                                          and len(decimals.lstrip("0")) <= 2), (sid, number)


def test_wrap_prints_bearings_that_subtract_to_the_printed_difference(progress, tmp_path, capsys):
    out = run(capsys, "run", "ekf-08-wrap", "--out", str(tmp_path))
    value = {key: float(re.search(rf"^\s+{key}\s+(-?[\d.]+)$", out, re.M).group(1))
             for key in ("z_theta", "z_hat_theta", "raw", "wrapped")}
    assert value == {"z_theta": -3.1279, "z_hat_theta": 3.1266, "raw": -6.2545, "wrapped": 0.0287}
    assert round(value["z_theta"] - value["z_hat_theta"], 4) == value["raw"]         # with 3 decimals: -6.255, not -6.254
    assert round(value["raw"] + 2 * math.pi, 4) == value["wrapped"]
    out = run(capsys, "run", "ekf-08-wrap", "--lang", "tr", "--out", str(tmp_path))
    assert re.search(r"^\s+raw\s+-6,2545$", out, re.M)


def test_run_keeps_two_significant_digits_of_small_values(progress, tmp_path, capsys):
    out = run(capsys, "run", "ekf-05-radar-update", "--out", str(tmp_path))
    assert re.search(r"^\s+H\s+\[\[0\.6, 0\.8\], \[-0\.0016, 0\.0012\]\]$", out, re.M)
    assert re.search(r"^\s+S\s+\[\[125, 0\], \[0, 0\.0008\]\]$", out, re.M)           # 1e-17 off the diagonal is 0
    out = run(capsys, "run", "ekf-05-radar-update", "--lang", "tr", "--out", str(tmp_path))
    assert "[[0,6; 0,8]; [-0,0016; 0,0012]]" in out


def test_run_matrix_and_none_values_print_readably(progress, tmp_path, capsys):
    out = run(capsys, "run", "08-covariance", "--out", str(tmp_path))
    assert "[[5, 1.5], [1.5, 0.5]]" in out                       # P_after_formula, F P F^T
    out = run(capsys, "run", "08-covariance", "--lang", "tr", "--out", str(tmp_path))
    assert "[[5; 1,5]; [1,5; 0,5]]" in out                       # no "1,5, 0,5" to misread
    out = run(capsys, "run", "ekf-08-wrap", "--seed", "13", "--out", str(tmp_path))    # no jump: nothing to print
    assert re.search(r"^\s+jump_scan\s+none$", out, re.M) and re.search(r"^\s+raw\s+none$", out, re.M)
    out = run(capsys, "run", "ekf-08-wrap", "--seed", "13", "--lang", "tr", "--out", str(tmp_path))
    assert re.search(r"^\s+jump_scan\s+yok$", out, re.M)


def test_run_options_reach_the_simulation(progress, tmp_path, capsys):
    out = run(capsys, "run", "03-1d-loop", "--q", "2", "--r", "3", "--out", str(tmp_path))
    assert re.search(r"^\s+Q\s+2$", out, re.M) and re.search(r"^\s+R\s+3$", out, re.M)
    out = run(capsys, "run", "14-nis", "--gap", "20:27", "--seed", "2", "--out", str(tmp_path))
    assert re.search(r"^\s+n\s+392$", out, re.M)
    out = run(capsys, "run", "14-nis", "--out", str(tmp_path))                  # no --gap: no gap
    assert re.search(r"^\s+n\s+400$", out, re.M)
    out = run(capsys, "run", "ekf-07-flyby", "--sigma-r", "2", "--sigma-theta", "0.01", "--q", "0.5",
              "--out", str(tmp_path))
    assert re.search(r"^\s+sigma_r\s+2$", out, re.M) and re.search(r"^\s+sigma_theta\s+0\.01$", out, re.M)
    assert re.search(r"^\s+q\s+0\.5$", out, re.M)
    out = run(capsys, "run", "ekf-03-range-only", "--sigma-r", "4", "--out", str(tmp_path))
    assert re.search(r"^\s+S\s+52$", out, re.M)                                 # 0.6² 100 + 4²


def test_run_ignored_options_are_named_on_stderr(progress, tmp_path, capsys):
    main(["run", "01-fusion", "--q", "2", "--seed", "3", "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert "01-fusion" in err and "--q" in err and "--seed" in err and "--r" not in err
    main(["run", "03-1d-loop", "--q", "2", "--out", str(tmp_path)])
    assert capsys.readouterr().err == ""
    main(["run", "14-nis", "--sigma-r", "2", "--sigma-theta", "0.1", "--out", str(tmp_path)])
    assert capsys.readouterr().err == "note: 14-nis does not use --sigma-r, --sigma-theta\n"
    main(["run", "ekf-03-range-only", "--r", "2", "--gap", "3:4", "--sigma-theta", "0.1", "--sigma-r", "3",
          "--out", str(tmp_path)])
    assert capsys.readouterr().err == "note: ekf-03-range-only does not use --r, --gap, --sigma-theta\n"
    main(["run", "ekf-02-tangent", "--seed", "1", "--lang", "tr", "--out", str(tmp_path)])
    assert capsys.readouterr().err == "not: ekf-02-tangent şu seçenekleri kullanmaz: --seed\n"
    main(["run", "ekf-08-wrap", "--q", "0.2", "--sigma-r", "3", "--sigma-theta", "0.01", "--seed", "1",
          "--out", str(tmp_path)])
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("argv", [
    ["run", "99-nope"], ["run", "03-1d-loop", "--gap", "5"], ["run", "03-1d-loop", "--gap", "9:3"],
    ["run", "03-1d-loop", "--gap", "0:3"], ["run", "03-1d-loop", "--gap", "none"], ["run", "03-1d-loop", "--r", "0"], ["run", "03-1d-loop", "--q", "-1"],
    ["run", "03-1d-loop", "--seed", "-1"], ["run", "03-1d-loop", "--lang", "de"], ["run"], ["nothing"], [],
    ["run", "ekf-07-flyby", "--sigma-r", "0"], ["run", "ekf-07-flyby", "--sigma-r", "-5"],
    ["run", "ekf-07-flyby", "--sigma-r", "nan"], ["run", "ekf-07-flyby", "--sigma-theta", "0"],
    ["run", "ekf-07-flyby", "--sigma-theta", "inf"], ["run", "ekf-07-flyby", "--sigma-theta", "1deg"],
    ["export", "x", "--lesson", "3"], ["export", "x", "--lesson", "two"], ["export"],
])
def test_bad_arguments_exit_with_a_usage_error(argv, progress, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


# --- gaps that do not fit ----------------------------------------------------------------

@pytest.mark.parametrize("sid, gap", [
    ("14-nis", "195:230"),          # hides the whole maneuver window 200..219: mean of an empty slice before
    ("14-nis", "1:400"),            # no measurement at all
    ("14-nis", "390:410"),          # beyond the 400 frames
    ("14-nis", "10:400"),           # fewer than the 20 measurements the 20-frame average needs
    ("13-q-r-tradeoff", "1:60"),    # no measurement at all
    ("13-q-r-tradeoff", "50:70"),   # beyond the 60 frames
])
def test_gaps_that_do_not_fit_are_usage_errors_without_traceback_or_warning(sid, gap, progress, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:                    # pytest runs with -W error: a RuntimeWarning would fail here
        main(["run", sid, "--gap", gap, "--out", str(tmp_path)])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "error" in err and sid in err and gap.split(":")[0] in err and "Traceback" not in err
    assert not (tmp_path / f"{sid}.png").exists() and not progress.exists()


@pytest.mark.parametrize("sid, gap", [
    ("14-nis", "1:199"), ("14-nis", "230:400"), ("14-nis", "100:205"), ("14-nis", "381:400"), ("14-nis", "1:1"),
    ("13-q-r-tradeoff", "1:59"), ("13-q-r-tradeoff", "36:60"), ("13-q-r-tradeoff", "20:27"),
])
def test_gaps_that_fit_run_without_warnings(sid, gap, progress, tmp_path, capsys):
    run(capsys, "run", sid, "--gap", gap, "--out", str(tmp_path))
    assert (tmp_path / f"{sid}.png").exists()


def test_a_gap_on_a_simulation_that_has_no_use_for_it_is_only_noted(progress, tmp_path, capsys):
    main(["run", "03-1d-loop", "--gap", "195:230", "--out", str(tmp_path)])
    assert "--gap" in capsys.readouterr().err


@pytest.mark.parametrize("options", [{"q": 0.0}, {"r": 1e-9}, {"q": 1e6}, {"r": 1e9}, {"seed": 99999999999}])
def test_extreme_but_valid_options_never_crash_a_simulation(options):
    readers = [e.id for e in registry.entries() if set(options) <= set(e.options)]
    assert readers, options                                  # the others do not read it (the test below shows that)
    for sid in readers:                                      # -W error: no numpy warning either
        res = sims.SIMS[sid](sims.Options(**options))
        assert res.summary and res.table, (sid, options)


@pytest.mark.parametrize("sid", [e.id for e in registry.entries() if {"sigma_r", "sigma_theta"} & set(e.options)])
@pytest.mark.parametrize("options", [{"sigma_r": 1e-9}, {"sigma_r": 1e6}, {"sigma_theta": 1e-9}, {"sigma_theta": 100.0},
                                     {"sigma_r": 1e-6, "sigma_theta": 1e-6}])
def test_extreme_radar_noise_never_crashes_a_simulation_or_its_figure(sid, options, tmp_path):
    res = sims.SIMS[sid](sims.Options(**options))            # the other simulations are shown not to read these
    assert res.summary and res.table
    assert all(math.isfinite(v) for v in _floats(res.summary)), (sid, options)
    assert res.plot(tmp_path).stat().st_size > 5000


def _floats(obj):
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _floats(v)


def test_option_error_is_what_the_simulations_raise():
    assert issubclass(sims.OptionError, ValueError)
    with pytest.raises(sims.OptionError, match="195:230"):
        sims.SIMS["14-nis"](sims.Options(gap=(195, 230)))
    with pytest.raises(sims.OptionError, match="60"):
        sims.SIMS["13-q-r-tradeoff"](sims.Options(gap=(50, 70)))


# --- what a learner sees after a run -----------------------------------------------------

def test_run_ends_with_the_essay_section_and_the_next_simulation(progress, tmp_path, capsys):
    out = run(capsys, "run", "03-1d-loop", "--out", str(tmp_path))
    tail = out.splitlines()[-3:]
    assert tail == ["In the essay: 3. The one-dimensional loop",
                    f"  {ESSAY}#3-the-one-dimensional-loop",
                    "Next: 01-fusion (fuse two measurements)"]
    out = run(capsys, "run", "14-nis", "--out", str(tmp_path))
    assert f"  {ESSAY}#14-is-the-filter-honest-the-nis-test" in out
    assert out.splitlines()[-1] == "Next: 01-fusion (fuse two measurements)"
    out = run(capsys, "run", "ekf-07-flyby", "--out", str(tmp_path))
    assert out.splitlines()[-3:] == ["In the essay: 7. Python from scratch", f"  {ESSAY2}#7-python-from-scratch",
                                     "Next: ekf-02-tangent (a belief through a curve, and its tangent)"]


def test_run_next_is_the_same_as_where_and_the_last_run_says_lesson_two_is_done(progress, tmp_path, capsys):
    ids = list(sims.SIMS)
    assert ids == LESSON1 + LESSON2
    for i, sid in enumerate(ids):
        e = registry.entry(sid)
        lines = run(capsys, "run", sid, "--out", str(tmp_path)).splitlines()
        assert lines[-3:-1] == [f"In the essay: {e.heading}",
                                f"  {registry.lesson_of(sid).essay[0]}#{registry.anchor(e.heading)}"]
        if i < len(ids) - 1:
            nxt = f"Next: {ids[i + 1]} ({registry.entry(ids[i + 1]).title[0]})"
            assert lines[-1] == (f"Lesson 1 is done. {nxt}" if sid == "14-nis" else nxt)
            assert run(capsys, "where").endswith(f" {lines[-1]}.\n")
    assert lines[-1] == "Lesson 2 is done."


def test_turkish_run_points_to_the_turkish_section_and_says_sirada(progress, tmp_path, capsys):
    out = run(capsys, "run", "03-1d-loop", "--lang", "tr", "--out", str(tmp_path))
    assert out.splitlines()[-3:] == ["Yazıda: 3. Tek boyutlu döngü", f"  {ESSAY}-tr#3-tek-boyutlu-dng",
                                     "Sırada: 01-fusion (iki ölçümü birleştir)"]
    assert "Next" not in out and "In the essay" not in out


@pytest.mark.parametrize("e", registry.entries(), ids=lambda e: e.id)
def test_every_turkish_run_links_its_turkish_heading(e, progress, tmp_path, capsys):
    lines = run(capsys, "run", e.id, "--lang", "tr", "--out", str(tmp_path)).splitlines()
    i = lines.index(f"Yazıda: {e.heading_tr}")
    assert lines[i + 1] == f"  {registry.lesson_of(e.id).essay[1]}#{registry.anchor(e.heading_tr)}"
    assert lines[i + 1].startswith(f"  {ESSAY}-tr#" if e.id in LESSON1 else f"  {ESSAY2}-tr#")


# --- Turkish -----------------------------------------------------------------------------

def test_turkish_list_where_and_run(progress, tmp_path, capsys):
    lines = run(capsys, "list", "--lang", "tr").splitlines()
    assert lines[0] == "DERS 1 · Kalman filtresi" and lines[8] == "DERS 2 · EKF"
    assert lines[-1].endswith("ekf-09-consistency       teğet nerede yetmez: dört durum")
    assert lines[1].startswith("  ▶ 01-fusion") and lines[1].endswith("iki ölçümü birleştir")
    out = run(capsys, "run", "05-derivative-vs-filter", "--lang", "tr", "--out", str(tmp_path))
    assert "yazıldı" in out and "wrote" not in out
    assert re.search(r"^\s+std_filter\s+0,\d+$", out, re.M)               # decimal comma, as on the figures
    assert re.search(r"^\s+theory_derivative\s+4,243$", out, re.M)
    where = run(capsys, "where", "--lang", "tr")
    assert where.startswith("Ders 1 ·") and "Sırada: 01-fusion (iki ölçümü birleştir)" in where and "Next" not in where
    assert (tmp_path / "05-derivative-vs-filter.png").exists()


def test_turkish_and_english_figures_differ_but_share_a_file_name(progress, tmp_path, capsys):
    run(capsys, "run", "01-fusion", "--out", str(tmp_path / "en"))
    run(capsys, "run", "01-fusion", "--lang", "tr", "--out", str(tmp_path / "tr"))
    assert (tmp_path / "en" / "01-fusion.png").read_bytes() != (tmp_path / "tr" / "01-fusion.png").read_bytes()


# --- registry ----------------------------------------------------------------------------

def test_registry_lists_exactly_the_simulations():
    assert [e.id for e in registry.entries()] == list(sims.SIMS)
    assert [lesson.number for lesson in registry.LESSONS] == [1, 2]                # lesson 3 is not announced yet
    assert [e.id for e in registry.LESSONS[0].entries] == LESSON1
    assert [e.id for e in registry.LESSONS[1].entries] == LESSON2 == list(sims2.SIMS)
    assert registry.LESSONS[0].essay == (ESSAY, ESSAY + "-tr") and registry.LESSONS[1].essay == (ESSAY2, ESSAY2 + "-tr")
    assert registry.LESSONS[1].name == ("EKF", "EKF")
    for e in registry.entries():
        assert e.title[0] and e.title[1] and e.title[0] != e.title[1]
        assert registry.lesson_of(e.id).essay[0] == (ESSAY if e.id in LESSON1 else ESSAY2)
        assert set(e.options) <= {"q", "r", "gap", "seed", "sigma_r", "sigma_theta"}


def test_anchor_follows_the_site_rule():
    assert registry.anchor("1. Fusing two measurements") == "1-fusing-two-measurements"
    assert registry.anchor("14. Is the filter honest? The NIS test") == "14-is-the-filter-honest-the-nis-test"
    assert registry.anchor("8. The covariance matrix P") == "8-the-covariance-matrix-p"
    assert registry.anchor("5. One radar update by hand") == "5-one-radar-update-by-hand"
    assert registry.anchor("a  -- b_c") == "a-b_c"


def test_anchor_drops_turkish_letters_like_the_site():
    assert registry.anchor("1. İki ölçümü birleştirmek") == "1-iki-lm-birletirmek"      # İ -> i, the rest dropped
    assert registry.anchor("8. Açı başa sarınca") == "8-a-baa-sarnca"
    assert registry.anchor("9. Teğet nerede yetmez?") == "9-teet-nerede-yetmez"
    assert registry.anchor("5. Hız neden durumun içinde olmalı?") == "5-hz-neden-durumun-iinde-olmal"
    assert registry.anchor("13. Q ve R’yi ayarlamak") == "13-q-ve-ryi-ayarlamak"
    assert registry.anchor("14. Filtre dürüst mü? NIS testi") == "14-filtre-drst-m-nis-testi"


def test_turkish_headings_have_the_english_section_numbers():
    for e in registry.entries():
        assert e.heading_tr.split(".")[0] == e.heading.split(".")[0], e.id
        assert e.heading_tr != e.heading


@pytest.mark.parametrize("sid", list(sims.SIMS))
def test_declared_options_are_exactly_the_ones_that_change_the_result(sid):
    base = sims.SIMS[sid](sims.Options()).summary
    probes = {"q": 0.5, "r": 2.0, "seed": 3, "gap": (20, 27), "sigma_r": 3.0, "sigma_theta": 0.03}
    declared = registry.entry(sid).options
    for name in declared:                                    # each one it reads changes the result on its own
        assert sims.SIMS[sid](sims.Options(**{name: probes[name]})).summary != base, (sid, name)
    unread = {name: value for name, value in probes.items() if name not in declared}
    assert sims.SIMS[sid](sims.Options(**unread)).summary == base, (sid, unread)   # and all the others together none


# --- the READMEs say what the code says --------------------------------------------------

def test_readme_table_links_every_id_to_its_essay_section():
    readme = (ROOT / "README.md").read_text()
    for lesson, heading in zip(registry.LESSONS, ("## Lesson 1: Kalman filter", "## Lesson 2: extended Kalman filter")):
        table = readme.split(heading + "\n")[1].split("\n## ")[0]
        for e in lesson.entries:
            row = next(line for line in table.splitlines() if line.startswith(f"| `{e.id}` |"))
            assert row == (f"| `{e.id}` | {e.title[0]} | [{e.heading}]({lesson.essay[0]}#{registry.anchor(e.heading)}) |")
        assert f"({lesson.essay[0]})" in readme and f"({lesson.essay[1]})" in readme      # both essays, both languages
    assert "is being written" not in readme
    for needle in ("git clone https://github.com/mftnakrsu/fusion-by-hand", "uv sync", "uv run fbh list",
                   "uv run fbh run 03-1d-loop", "MIT"):
        assert needle in readme


def test_readme_commands_are_all_written_with_uv_run_and_the_progress_file_is_documented():
    readme = (ROOT / "README.md").read_text()
    assert not re.search(r"^fbh ", readme, re.M)                               # a learner cannot type a bare `fbh`
    for cmd in ("list", "where", "run", "export", "export <dir> --lesson 2"):
        assert f"uv run fbh {cmd}" in readme
    assert ".fbh-progress.json" in readme and "FBH_PROGRESS" in readme and "current directory" in readme
    options = readme.split("Options of")[1].split("\n\n")[0]
    assert "none" not in options                                               # `--gap none` is gone
    for flag in ("--q", "--r", "--gap", "--sigma-r", "--sigma-theta", "--seed", "--lang", "--out"):
        assert f"`{flag}" in options, flag
    for e in registry.entries():                                               # who reads the radar's noise
        for name in ("sigma_r", "sigma_theta"):
            flag = "--" + name.replace("_", "-")
            said = options.split(f"`{flag}` is read by")[1].split(".")[0]
            assert (f"`{e.id}`" in said) == (name in e.options), (e.id, flag)


LESSON_DIRS = {1: "lesson-01-kalman-filter", 2: "lesson-02-extended-kalman-filter"}


def test_lesson_readme_has_a_paragraph_and_a_command_per_id():
    for lesson in registry.LESSONS:
        text = (ROOT / LESSON_DIRS[lesson.number] / "README.md").read_text()
        assert text.startswith(f"# Lesson {lesson.number}: ")
        for e in lesson.entries:
            assert f"\n## {e.id}: " in text and f"uv run fbh run {e.id}" in text, e.id
    readme = (ROOT / "README.md").read_text()
    for folder in LESSON_DIRS.values():
        assert f"]({folder}/README.md)" in readme


@pytest.mark.parametrize("sid", LESSON2)
def test_lesson_two_readme_quotes_what_the_cli_prints(sid, progress, tmp_path, capsys):
    """Every `key value` the section of `sid` quotes is a line `fbh run <id>` prints, with the default options."""
    text = (ROOT / LESSON_DIRS[2] / "README.md").read_text()
    section = text.split(f"\n## {sid}: ")[1].split("\n## ")[0]
    printed = dict(re.findall(r"^  (\S+)\s+(.+)$", run(capsys, "run", sid, "--out", str(tmp_path)).split("\nwrote ")[0], re.M))
    quoted = [(key, value) for key, value in re.findall(r"`(\w+) ([^`]+)`", section) if key in printed]
    assert len(quoted) >= 3, sid
    for key, value in quoted:
        assert printed[key] == value, (sid, key, value, printed[key])


def test_ci_workflow_runs_the_tests_with_uv():
    wf = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    for needle in ("actions/checkout@v4", "astral-sh/setup-uv@v6", "uv sync", "uv run pytest -q"):
        assert needle in wf
