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
        "where": "Lesson {n} · {done} of {total} simulations run. Next: {id} ({title}).",
        "where_done": "Lesson {n} · {done} of {total} simulations run. Lesson {n} is complete.",
        "wrote": "wrote", "ignored": "note: {id} does not use {flags}",
    },
    "tr": {
        "lesson": "DERS {n} · {name}", "writing": "(yazılıyor)", "none": "yok",
        "where": "Ders {n} · çalıştırılan simülasyon: {done} / {total}. Sıradaki: {id} ({title}).",
        "where_done": "Ders {n} · çalıştırılan simülasyon: {done} / {total}. Ders {n} tamam.",
        "wrote": "yazıldı:", "ignored": "not: {id} şu seçenekleri kullanmaz: {flags}",
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
    """'A:B' (frames A to B, 1-based, inclusive) or 'none'."""
    if text.lower() == "none":
        return None
    try:
        a, b = (int(part) for part in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected A:B (for example 20:27) or none, got {text!r}") from None
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
    text, i = _TEXT[args.lang], 0 if args.lang == "en" else 1
    lesson, n_done, nxt = _first_unrun(set(progress.load()))
    fields = {"n": lesson.number, "done": n_done, "total": len(lesson.entries)}
    if nxt is None:
        print(text["where_done"].format(**fields))
    else:
        print(text["where"].format(**fields, id=nxt.id, title=nxt.title[i]))


def cmd_run(args) -> None:
    text, i = _TEXT[args.lang], 0 if args.lang == "en" else 1
    entry = registry.entry(args.id)
    given = {"q": args.q, "r": args.r, "gap": args.gap, "seed": args.seed}
    unused = [f"--{name}" for name, value in given.items() if value is not None and name not in entry.options]
    if unused:
        print(text["ignored"].format(id=entry.id, flags=", ".join(unused)), file=sys.stderr)

    result = sims.SIMS[entry.id](sims.Options(q=args.q, r=args.r, gap=args.gap, seed=args.seed, lang=args.lang))
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
    run.add_argument("--gap", type=_gap, metavar="A:B|none", help="frames without a measurement, for example 20:27")
    run.add_argument("--seed", type=_seed, help="random seed")
    run.add_argument("--out", default="out", help="folder for <id>.png and <id>.csv (default out)")
    run.set_defaults(func=cmd_run)

    exp = sub.add_parser("export", help="write the JSON the essay's browser figures use")
    exp.add_argument("dir", help="folder to write tracking.json, kf-cases.json, derivative.json, covariance.json, nis.json to")
    exp.set_defaults(func=cmd_export)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
