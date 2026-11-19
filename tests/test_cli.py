import csv
import json
import re
from pathlib import Path

import pytest

from fbh import registry, sims
from fbh.cli import main

ROOT = Path(__file__).resolve().parent.parent
ESSAY = "https://www.mefby.com/essays/fusion-01-kalman-filter"


@pytest.fixture
def progress(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    monkeypatch.setenv("FBH_PROGRESS", str(path))
    return path


def run(capsys, *argv):
    code = main(list(argv))
    assert code in (0, None)
    return capsys.readouterr().out


# --- the brief's three commands ----------------------------------------------------------

def test_list_run_where_follow_the_brief(progress, tmp_path, capsys):
    out = run(capsys, "list")
    lines = out.splitlines()
    assert lines[0] == "LESSON 1 · Kalman filter" and lines[-1] == "LESSON 2 · EKF (writing)"
    assert len(lines) == 1 + 7 + 1
    assert lines[1] == "  ▶ 01-fusion                fuse two measurements"
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
    for sid in sims.SIMS:
        run(capsys, "run", sid, "--out", str(tmp_path))
    assert run(capsys, "where") == ("Lesson 1 · 7 of 7 simulations run. Lesson 1 is done. "
                                    "Lesson 2 (EKF) is being written.\n")
    assert run(capsys, "where", "--lang", "tr") == ("Ders 1 · çalıştırılan simülasyon: 7 / 7. Ders 1 bitti. "
                                                    "Ders 2 (EKF) yazılıyor.\n")
    assert "▶" not in run(capsys, "list") and run(capsys, "list").count("✓") == 7


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
        assert not re.search(r"\d\.\d{4,}", text), (sid, text)


def test_run_matrix_and_none_values_print_readably(progress, tmp_path, capsys):
    out = run(capsys, "run", "08-covariance", "--out", str(tmp_path))
    assert "[[5, 1.5], [1.5, 0.5]]" in out                       # P_after_formula, F P F^T
    out = run(capsys, "run", "08-covariance", "--lang", "tr", "--out", str(tmp_path))
    assert "[[5; 1,5]; [1,5; 0,5]]" in out                       # no "1,5, 0,5" to misread


def test_run_options_reach_the_simulation(progress, tmp_path, capsys):
    out = run(capsys, "run", "03-1d-loop", "--q", "2", "--r", "3", "--out", str(tmp_path))
    assert re.search(r"^\s+Q\s+2$", out, re.M) and re.search(r"^\s+R\s+3$", out, re.M)
    out = run(capsys, "run", "14-nis", "--gap", "20:27", "--seed", "2", "--out", str(tmp_path))
    assert re.search(r"^\s+n\s+392$", out, re.M)
    out = run(capsys, "run", "14-nis", "--out", str(tmp_path))                  # no --gap: no gap
    assert re.search(r"^\s+n\s+400$", out, re.M)


def test_run_ignored_options_are_named_on_stderr(progress, tmp_path, capsys):
    main(["run", "01-fusion", "--q", "2", "--seed", "3", "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert "01-fusion" in err and "--q" in err and "--seed" in err and "--r" not in err
    main(["run", "03-1d-loop", "--q", "2", "--out", str(tmp_path)])
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("argv", [
    ["run", "99-nope"], ["run", "03-1d-loop", "--gap", "5"], ["run", "03-1d-loop", "--gap", "9:3"],
    ["run", "03-1d-loop", "--gap", "0:3"], ["run", "03-1d-loop", "--gap", "none"], ["run", "03-1d-loop", "--r", "0"], ["run", "03-1d-loop", "--q", "-1"],
    ["run", "03-1d-loop", "--seed", "-1"], ["run", "03-1d-loop", "--lang", "de"], ["run"], ["nothing"], [],
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
    for sid, sim in sims.SIMS.items():                       # -W error: no numpy warning either
        res = sim(sims.Options(**options))
        assert res.summary and res.table, (sid, options)


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


def test_run_next_is_the_same_as_where_and_the_last_run_says_lesson_one_is_done(progress, tmp_path, capsys):
    ids = list(sims.SIMS)
    for i, sid in enumerate(ids):
        out = run(capsys, "run", sid, "--out", str(tmp_path))
        if i < len(ids) - 1:
            assert out.splitlines()[-1] == f"Next: {ids[i + 1]} ({registry.entry(ids[i + 1]).title[0]})"
    assert out.splitlines()[-1] == "Lesson 1 is done. Lesson 2 (EKF) is being written."
    assert "Next:" not in out


def test_turkish_run_points_to_the_turkish_essay_and_says_sirada(progress, tmp_path, capsys):
    out = run(capsys, "run", "03-1d-loop", "--lang", "tr", "--out", str(tmp_path))
    assert out.splitlines()[-3:] == ["Yazıda: bölüm 3", f"  {ESSAY}-tr",
                                     "Sırada: 01-fusion (iki ölçümü birleştir)"]
    assert "Next" not in out and "In the essay" not in out


# --- Turkish -----------------------------------------------------------------------------

def test_turkish_list_where_and_run(progress, tmp_path, capsys):
    lines = run(capsys, "list", "--lang", "tr").splitlines()
    assert lines[0] == "DERS 1 · Kalman filtresi" and lines[-1] == "DERS 2 · EKF (yazılıyor)"
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
    assert [lesson.number for lesson in registry.LESSONS] == [1, 2]
    assert registry.LESSONS[1].entries == ()                                  # lesson 2 is still being written
    for e in registry.entries():
        assert e.title[0] and e.title[1] and e.title[0] != e.title[1]


def test_anchor_follows_the_site_rule():
    assert registry.anchor("1. Fusing two measurements") == "1-fusing-two-measurements"
    assert registry.anchor("14. Is the filter honest? The NIS test") == "14-is-the-filter-honest-the-nis-test"
    assert registry.anchor("8. The covariance matrix P") == "8-the-covariance-matrix-p"
    assert registry.anchor("a  -- b_c") == "a-b_c"


@pytest.mark.parametrize("sid", list(sims.SIMS))
def test_declared_options_are_exactly_the_ones_that_change_the_result(sid):
    base = sims.SIMS[sid](sims.Options()).summary
    probes = {"q": 0.5, "r": 2.0, "seed": 3, "gap": (20, 27)}
    for name, value in probes.items():
        changed = sims.SIMS[sid](sims.Options(**{name: value})).summary != base
        assert changed == (name in registry.entry(sid).options), (sid, name)


# --- the READMEs say what the code says --------------------------------------------------

def test_readme_table_links_every_id_to_its_essay_section():
    readme = (ROOT / "README.md").read_text()
    for e in registry.entries():
        assert f"`{e.id}`" in readme
        assert f"{ESSAY}#{registry.anchor(e.heading)}" in readme, e.id
    assert registry.ESSAY_URL == ESSAY
    for needle in ("git clone https://github.com/mftnakrsu/fusion-by-hand", "uv sync", "uv run fbh list",
                   "uv run fbh run 03-1d-loop", "MIT"):
        assert needle in readme


def test_readme_commands_are_all_written_with_uv_run_and_the_progress_file_is_documented():
    readme = (ROOT / "README.md").read_text()
    assert not re.search(r"^fbh ", readme, re.M)                               # a learner cannot type a bare `fbh`
    for cmd in ("list", "where", "run", "export"):
        assert f"uv run fbh {cmd}" in readme
    assert ".fbh-progress.json" in readme and "FBH_PROGRESS" in readme and "current directory" in readme
    assert "none" not in readme.split("Options of")[1].split("\n\n")[0]       # `--gap none` is gone


def test_lesson_readme_has_a_paragraph_and_a_command_per_id():
    text = (ROOT / "lesson-01-kalman-filter" / "README.md").read_text()
    for e in registry.entries():
        assert f"uv run fbh run {e.id}" in text, e.id


def test_ci_workflow_runs_the_tests_with_uv():
    wf = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    for needle in ("actions/checkout@v4", "astral-sh/setup-uv@v6", "uv sync", "uv run pytest -q"):
        assert needle in wf
