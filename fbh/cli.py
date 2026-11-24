"""The `fbh` command: list the simulations, say where you are, run one, export the figure data."""
import argparse
import csv
import math
import sys
from pathlib import Path

from . import export, export2, progress, registry, sims

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
        "essay": "Yazıda: {heading}",
    },
}
LANGS = tuple(_TEXT)
EXPORTS = {1: export, 2: export2}                           # `fbh export --lesson n` writes EXPORTS[n].FILES


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


def _sigma_r(text):
    return _float(text, "sigma-r must be more than 0", lambda v: v > 0)


def _sigma_theta(text):
    return _float(text, "sigma-theta must be more than 0", lambda v: v > 0)


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

def _fmt(value, lang: str, decimals: int = 3) -> str:
    """A summary value for the terminal: at most `decimals` decimals, no trailing zeros, comma in Turkish.

    A value under 0.01 keeps two significant digits instead (a Jacobian's bearing row reads [-0.0016, 0.0012],
    not [-0.002, 0.001]); one under 1e-9 is rounding noise and prints 0.
    """
    if value is None:
        return _TEXT[lang]["none"]
    if isinstance(value, float):
        digits = 1 - math.floor(math.log10(abs(value))) if 1e-9 <= abs(value) < 0.01 else decimals
        s = f"{round(value, digits):.{digits}f}".rstrip("0").rstrip(".")
        s = "0" if s in ("", "-0") else s
        return s.replace(".", ",") if lang == "tr" else s
    if isinstance(value, (list, tuple)):                    # Turkish: "; " so that "2,5, 1" cannot be misread
        return "[" + ("; " if lang == "tr" else ", ").join(_fmt(v, lang, decimals) for v in value) + "]"
    return str(value)


def _print_table(table: list[dict], lang: str, digits: dict) -> None:
    """A short table on the terminal, above the summary: a header, columns two spaces apart, then a blank line.
    Numbers print as in the summary; None is an empty cell, as in the CSV."""
    cols = list(table[0])
    cells = [cols] + [["" if row[c] is None else _fmt(row[c], lang, digits.get(c, 3)) for c in cols] for row in table]
    widths = [max(len(line[j]) for line in cells) for j in range(len(cols))]
    for line in cells:
        print("  " + "  ".join(cell.ljust(w) for cell, w in zip(line, widths)).rstrip())
    print()


def _write_csv(table: list[dict], path: Path) -> None:
    """The table as CSV: floats to 12 significant digits (so 0.8999999999999998 reads 0.9), None as an empty cell."""
    cell = lambda v: f"{v:.12g}" if isinstance(v, float) else v
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: cell(v) for k, v in row.items()} for row in table)


def _current_lesson(run: list[str]):
    """The lesson you are in: the one of the last simulation you ran for the first time (lesson 1 before any)."""
    known = [sid for sid in run if sid in {e.id for e in registry.entries()}]
    return registry.lesson_of(known[-1]) if known else registry.LESSONS[0]


def _next(lesson, done: set[str]):
    """What comes after a simulation of `lesson`: its first one not run yet; once it is done, the first one not
    run in a later lesson; None when this lesson and every later one are done."""
    later = [e for l in registry.LESSONS if l.number >= lesson.number for e in l.entries]
    return next((e for e in later if e.id not in done), None)


def _next_line(lang: str, lesson, done: set[str]) -> str:
    """"Next: <id> (<title>)" while `lesson` has a simulation left; then that it is done and what comes after it."""
    text, i = _TEXT[lang], 0 if lang == "en" else 1
    nxt = _next(lesson, done)
    if nxt is not None and nxt in lesson.entries:
        return text["next"].format(id=nxt.id, title=nxt.title[i])
    line = text["done"].format(n=lesson.number)
    if nxt is not None:
        return f"{line} {text['next'].format(id=nxt.id, title=nxt.title[i])}"
    coming = [l for l in registry.LESSONS if l.number > lesson.number and not l.entries]
    return line + (" " + text["coming"].format(n=coming[0].number, name=coming[0].name[i]) if coming else "")


# --- commands ----------------------------------------------------------------------------

def cmd_list(args) -> None:
    text, i = _TEXT[args.lang], 0 if args.lang == "en" else 1
    run = progress.load()
    done = set(run)
    nxt = _next(_current_lesson(run), done)
    width = max(len(e.id) for e in registry.entries())
    for lesson in registry.LESSONS:
        head = text["lesson"].format(n=lesson.number, name=lesson.name[i])
        print(head if lesson.entries else f"{head} {text['writing']}")
        for e in lesson.entries:
            mark = "✓" if e.id in done else "▶" if e == nxt else " "
            print(f"  {mark} {e.id:<{width}}  {e.title[i]}")


def cmd_where(args) -> None:
    text = _TEXT[args.lang]
    run = progress.load()
    lesson, done = _current_lesson(run), set(run)
    head = text["progress"].format(n=lesson.number, done=sum(e.id in done for e in lesson.entries),
                                   total=len(lesson.entries))
    line = _next_line(args.lang, lesson, done)
    print(f"{head} {line}" + ("" if line.endswith(".") else "."))


def cmd_run(args) -> None:
    text, i = _TEXT[args.lang], 0 if args.lang == "en" else 1
    entry = registry.entry(args.id)
    given = {"q": args.q, "r": args.r, "gap": args.gap, "seed": args.seed, "sigma_r": args.sigma_r,
             "sigma_theta": args.sigma_theta}
    unused = [f"--{name.replace('_', '-')}" for name, value in given.items()
              if value is not None and name not in entry.options]
    if unused:
        print(text["ignored"].format(id=entry.id, flags=", ".join(unused)), file=sys.stderr)

    try:
        result = sims.SIMS[entry.id](sims.Options(**given, lang=args.lang))
    except sims.OptionError as e:
        raise UsageError(f"{entry.id}: {e}") from None
    out = Path(args.out)
    png = result.plot(out, args.lang)
    table = out / f"{entry.id}.csv"
    _write_csv(result.table, table)
    progress.mark(entry.id)

    print(f"{entry.id} · {entry.title[i]}")
    if result.show_table:
        _print_table(result.table, args.lang, result.digits)
    width = max(len(key) for key in result.summary)
    for key, value in result.summary.items():
        print(f"  {key:<{width}}  {_fmt(value, args.lang, result.digits.get(key, 3))}")
    print(f"{text['wrote']} {png}, {table}")
    heading = (entry.heading, entry.heading_tr)[i]
    print(text["essay"].format(heading=heading))
    print(f"  {registry.lesson_of(entry.id).essay[i]}#{registry.anchor(heading)}")
    print(_next_line(args.lang, registry.lesson_of(entry.id), set(progress.load())))


def cmd_export(args) -> None:
    for path in EXPORTS[args.lesson].write_all(args.dir):
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
    run.add_argument("--sigma-r", type=_sigma_r, help="the radar's range noise, in metres")
    run.add_argument("--sigma-theta", type=_sigma_theta, help="the radar's bearing noise, in radians")
    run.add_argument("--seed", type=_seed, help="random seed")
    run.add_argument("--out", default="out", help="folder for <id>.png and <id>.csv (default out)")
    run.set_defaults(func=cmd_run)

    exp = sub.add_parser("export", help="write the JSON the essay's browser figures use")
    exp.add_argument("dir", help="folder to write the files to (created if needed)")
    exp.add_argument("--lesson", type=int, choices=list(EXPORTS), default=1,
                     help="; ".join(f"{n} writes {', '.join(module.FILES)}" for n, module in EXPORTS.items())
                     + " (default 1)")
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
