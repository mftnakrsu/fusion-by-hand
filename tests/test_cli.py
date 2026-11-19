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
    done = run(capsys, "where")
    assert done.startswith("Lesson 1 · 7 of 7 simulations run.") and "Next:" not in done and "complete" in done
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


def test_a_broken_progress_file_counts_as_nothing_run(progress, capsys):
    progress.write_text("{ not json")
    assert "0 of 7" in run(capsys, "where")


def test_a_failed_run_is_not_recorded(progress, tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        main(["run", "01-fusion", "--out", str(blocker / "sub")])
    assert not progress.exists() or "01-fusion" not in progress.read_text()


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
    out = run(capsys, "run", "14-nis", "--gap", "none", "--out", str(tmp_path))
    assert re.search(r"^\s+n\s+400$", out, re.M)


def test_run_ignored_options_are_named_on_stderr(progress, tmp_path, capsys):
    main(["run", "01-fusion", "--q", "2", "--seed", "3", "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert "01-fusion" in err and "--q" in err and "--seed" in err and "--r" not in err
    main(["run", "03-1d-loop", "--q", "2", "--out", str(tmp_path)])
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("argv", [
    ["run", "99-nope"], ["run", "03-1d-loop", "--gap", "5"], ["run", "03-1d-loop", "--gap", "9:3"],
    ["run", "03-1d-loop", "--gap", "0:3"], ["run", "03-1d-loop", "--r", "0"], ["run", "03-1d-loop", "--q", "-1"],
    ["run", "03-1d-loop", "--seed", "-1"], ["run", "03-1d-loop", "--lang", "de"], ["run"], ["nothing"], [],
])
def test_bad_arguments_exit_with_a_usage_error(argv, progress, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


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
    assert where.startswith("Ders 1 ·") and "Sıradaki: 01-fusion (iki ölçümü birleştir)" in where and "Next" not in where
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


def test_lesson_readme_has_a_paragraph_and_a_command_per_id():
    text = (ROOT / "lesson-01-kalman-filter" / "README.md").read_text()
    for e in registry.entries():
        assert f"uv run fbh run {e.id}" in text, e.id


def test_ci_workflow_runs_the_tests_with_uv():
    wf = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    for needle in ("actions/checkout@v4", "astral-sh/setup-uv@v6", "uv sync", "uv run pytest -q"):
        assert needle in wf
