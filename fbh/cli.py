"""The `fbh` command: list the simulations, say where you are, run one, export the figure data."""
import argparse
import csv
import math
import sys
from pathlib import Path

from . import export, progress, registry, sims

_TEXT = {
    "en": {
        "lesson": "LESSON {n} · {name}", "writing": "(writing)", "none": "none",
        "progress": "Lesson {n} · {done} of {total} simulations run.",
        "next": "Next: {id} ({title})", "done": "Lesson {n} is done.", "coming": "Lesson {n} ({name}) is being written.",
        "wrote": "wrote", "ignored": "note: {id} does not use {flags}",
        "essay": "In the essay: {heading}",
    },
    "tr": {
        "lesson": "DERS {n} · {name}", "writing": "(yazılıyor)", "none": "yok",
        "progress": "Ders {n} · çalıştırılan simülasyon: {done} / {total}.",
        "next": "Sırada: {id} ({title})", "done": "Ders {n} bitti.", "coming": "Ders {n} ({name}) yazılıyor.",
        "wrote": "yazıldı:", "ignored": "not: {id} şu seçenekleri kullanmaz: {flags}",
        "essay": "Yazıda: bölüm {number}",
    },
}
LANGS = tuple(_TEXT)


# --- argument types ----------------------------------------------------------------------

def _float(text: str, name: str, ok) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or not ok(value):
        raise argparse.ArgumentTypeError(f"{name}, got {text}")
    return value


def _q(text):
    return _float(text, "q must be 0 or more", lambda v: v >= 0)


def _r(text):
    return _float(text, "r must be more than 0", lambda v: v > 0)


def _seed(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"seed must be 0 or more, got {text}")
    return value


def _gap(text):
    """'A:B': frames A to B, 1-based, inclusive. Whether it fits the run is for the simulation to say."""
    try:
        a, b = (int(part) for part in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected A:B (for example 20:27), got {text!r}") from None
    if not 1 <= a <= b:
        raise argparse.ArgumentTypeError(f"need 1 <= A <= B, got {text}")
    return (a, b)


# --- output ------------------------------------------------------------------------------

def _fmt(value, lang: str) -> str:
    """A summary value for the terminal: at most 3 decimals, no trailing zeros, comma in Turkish."""
    if value is None:
        return _TEXT[lang]["none"]
    if isinstance(value, float):
        s = f"{round(value, 3):.3f}".rstrip("0").rstrip(".")
        s = "0" if s in ("", "-0") else s
        return s.replace(".", ",") if lang == "tr" else s
    if isinstance(value, (list, tuple)):                    # Turkish: "; " so that "2,5, 1" cannot be misread
        return "[" + ("; " if lang == "tr" else ", ").join(_fmt(v, lang) for v in value) + "]"
    return str(value)


def _write_csv(table: list[dict], path: Path) -> None:
    """The table as CSV: floats to 12 significant digits (so 0.8999999999999998 reads 0.9), None as an empty cell."""
    cell = lambda v: f"{v:.12g}" if isinstance(v, float) else v
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: cell(v) for k, v in row.items()} for row in table)


def _first_unrun(done: set[str]):
    """(lesson, how many of it are run, the first one that is not or None) for the lesson you are in."""
    lessons = [lesson for lesson in registry.LESSONS if lesson.entries]
    for lesson in lessons:
        todo = [e for e in lesson.entries if e.id not in done]
        if todo:
            return lesson, len(lesson.entries) - len(todo), todo[0]
    return lessons[-1], len(lessons[-1].entries), None


def _next_line(lang: str, lesson, nxt) -> str:
    """"Next: <id> (<title>)", or, when the lesson is finished, that it is and what comes after it."""
    text, i = _TEXT[lang], 0 if lang == "en" else 1
    if nxt is not None:
        return text["next"].format(id=nxt.id, title=nxt.title[i])
    line = text["done"].format(n=lesson.number)
    coming = [l for l in registry.LESSONS if l.number > lesson.number and not l.entries]
    return line + (" " + text["coming"].format(n=coming[0].number, name=coming[0].name[i]) if coming else "")


# --- commands ----------------------------------------------------------------------------

def cmd_list(args) -> None:
    text, i = _TEXT[args.lang], 0 if args.lang == "en" else 1
    done = set(progress.load())
    nxt = next((e.id for e in registry.entries() if e.id not in done), None)
    width = max(len(e.id) for e in registry.entries())
    for lesson in registry.LESSONS:
        head = text["lesson"].format(n=lesson.number, name=lesson.name[i])
        print(head if lesson.entries else f"{head} {text['writing']}")
        for e in lesson.entries:
            mark = "✓" if e.id in done else "▶" if e.id == nxt else " "
            print(f"  {mark} {e.id:<{width}}  {e.title[i]}")


def cmd_where(args) -> None:
    text = _TEXT[args.lang]
    lesson, n_done, nxt = _first_unrun(set(progress.load()))
    head = text["progress"].format(n=lesson.number, done=n_done, total=len(lesson.entries))
    print(f"{head} {_next_line(args.lang, lesson, nxt)}" + ("." if nxt is not None else ""))


def cmd_run(args) -> None:
    text, i = _TEXT[args.lang], 0 if args.lang == "en" else 1
    entry = registry.entry(args.id)
    given = {"q": args.q, "r": args.r, "gap": args.gap, "seed": args.seed}
    unused = [f"--{name}" for name, value in given.items() if value is not None and name not in entry.options]
    if unused:
        print(text["ignored"].format(id=entry.id, flags=", ".join(unused)), file=sys.stderr)

    try:
        result = sims.SIMS[entry.id](sims.Options(q=args.q, r=args.r, gap=args.gap, seed=args.seed, lang=args.lang))
    except sims.OptionError as e:
        raise UsageError(f"{entry.id}: {e}") from None
    out = Path(args.out)
    png = result.plot(out, args.lang)
    table = out / f"{entry.id}.csv"
    _write_csv(result.table, table)
    progress.mark(entry.id)

    print(f"{entry.id} · {entry.title[i]}")
    width = max(len(key) for key in result.summary)
    for key, value in result.summary.items():
        print(f"  {key:<{width}}  {_fmt(value, args.lang)}")
    print(f"{text['wrote']} {png}, {table}")
    if args.lang == "en":                                   # the Turkish essay has its own headings: no anchor
        print(text["essay"].format(heading=entry.heading))
        print(f"  {registry.ESSAY_URL}#{registry.anchor(entry.heading)}")
    else:
        print(text["essay"].format(number=registry.section_number(entry)))
        print(f"  {registry.ESSAY_URL_TR}")
    lesson, _, nxt = _first_unrun(set(progress.load()))
    print(_next_line(args.lang, lesson, nxt))


def cmd_export(args) -> None:
    for path in export.write_all(args.dir):
        print(f"wrote {path} ({path.stat().st_size / 1000:.0f} kB)")


# --- parser ------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fbh", description="Sensor Fusion from First Principles: lesson code.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{list,where,run,export}")
    lang = argparse.ArgumentParser(add_help=False)
    lang.add_argument("--lang", choices=LANGS, default="en", help="language of the text and the figure (default en)")

    sub.add_parser("list", parents=[lang], help="the simulations, what you have run, what is next").set_defaults(func=cmd_list)
    sub.add_parser("where", parents=[lang], help="how far you are and what to run next").set_defaults(func=cmd_where)

    run = sub.add_parser("run", parents=[lang], help="run one simulation: figure, table, numbers")
    run.add_argument("id", choices=list(sims.SIMS), metavar="id", help="one of: " + ", ".join(sims.SIMS))
    run.add_argument("--q", type=_q, help="process noise Q (the simulation's own default if left out)")
    run.add_argument("--r", type=_r, help="measurement noise R")
    run.add_argument("--gap", type=_gap, metavar="A:B", help="frames A to B have no measurement, for example 20:27")
    run.add_argument("--seed", type=_seed, help="random seed")
    run.add_argument("--out", default="out", help="folder for <id>.png and <id>.csv (default out)")
    run.set_defaults(func=cmd_run)

    exp = sub.add_parser("export", help="write the JSON the essay's browser figures use")
    exp.add_argument("dir", help="folder to write tracking.json, kf-cases.json, derivative.json, covariance.json, nis.json to")
    exp.set_defaults(func=cmd_export)
    return parser


class UsageError(Exception):
    """A request that cannot be carried out; shown like an argparse error (one line, exit status 2)."""


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except UsageError as e:
        parser.error(str(e))
    except OSError as e:                                    # a folder that cannot be written, ...
        parser.error(f"{e.strerror}: {e.filename}" if e.filename else str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
