"""The lessons and their simulations: ids, short titles (EN, TR), essay sections and the options each one reads.

`list` and `where` are written from this, and a test keeps it in step with `sims.SIMS` and the README.
Lesson 2 is announced but has no simulations yet.
"""
import re
from dataclasses import dataclass

ESSAY_URL = "https://www.mefby.com/essays/fusion-01-kalman-filter"
ESSAY_URL_TR = ESSAY_URL + "-tr"        # the Turkish essay, with its own Turkish headings


@dataclass(frozen=True)
class Entry:
    id: str
    title: tuple[str, str]               # (English, Turkish), as printed by `fbh list`
    heading: str                         # the essay heading the simulation belongs to
    heading_tr: str                      # the same heading in the Turkish essay
    options: tuple[str, ...]             # which of --q --r --gap --seed the simulation reads


@dataclass(frozen=True)
class Lesson:
    number: int
    name: tuple[str, str]
    entries: tuple[Entry, ...]           # empty: still being written


LESSONS = (
    Lesson(1, ("Kalman filter", "Kalman filtresi"), (
        Entry("01-fusion", ("fuse two measurements", "iki ölçümü birleştir"),
              "1. Fusing two measurements", "1. İki ölçümü birleştirmek", ()),
        Entry("03-1d-loop", ("predict + update, a measurement gap", "tahmin + güncelleme, bir ölçüm boşluğu"),
              "3. The one-dimensional loop", "3. Tek boyutlu döngü", ("q", "r")),
        Entry("05-derivative-vs-filter", ("why velocity belongs in the state", "hız neden durumun içinde olmalı"),
              "5. Why velocity belongs in the state", "5. Hız neden durumun içinde olmalı?",
              ("q", "r", "seed")),
        Entry("08-covariance", ("the covariance matrix P, before and after F", "P kovaryans matrisi, F'den önce ve sonra"),
              "8. The covariance matrix P", "8. Kovaryans matrisi P", ("seed",)),
        Entry("12-velocity-learning", ("the filter learns a speed it never measures", "filtre hiç ölçmediği hızı öğrenir"),
              "12. Python from scratch", "12. Python ile sıfırdan", ("q", "r", "seed")),
        Entry("13-q-r-tradeoff", ("tuning Q and R: calm or agile", "Q ve R ayarı: sakin mi çevik mi"),
              "13. Tuning Q and R", "13. Q ve R’yi ayarlamak", ("q", "r", "gap", "seed")),
        Entry("14-nis", ("is the filter honest? the NIS test", "filtre dürüst mü? NIS testi"),
              "14. Is the filter honest? The NIS test", "14. Filtre dürüst mü? NIS testi",
              ("q", "r", "gap", "seed")),
    )),
    Lesson(2, ("EKF", "EKF"), ()),
)


def entries() -> list[Entry]:
    return [e for lesson in LESSONS for e in lesson.entries]


def entry(sid: str) -> Entry:
    return next(e for e in entries() if e.id == sid)


def anchor(heading: str) -> str:
    """The id the site gives a `##` heading: lowercase, only [a-z0-9_ -], spaces to '-', '-' runs collapsed.

    As on the site, letters outside a-z are dropped, so Turkish ı ş ğ ü ö ç vanish ("İ" lowercases to "i" plus a
    combining dot, and only the dot is dropped).
    """
    kept = re.sub(r"[^a-z0-9_ -]", "", heading.lower())
    return re.sub(r"-+", "-", kept.replace(" ", "-"))

