"""The lessons and their simulations: ids, short titles (EN, TR), essay sections and the options each one reads.

`list` and `where` are written from this, and a test keeps it in step with `sims.SIMS` and the README.
"""
import re
from dataclasses import dataclass

ESSAYS = "https://www.mefby.com/essays/"


@dataclass(frozen=True)
class Entry:
    id: str
    title: tuple[str, str]               # (English, Turkish), as printed by `fbh list`
    heading: str                         # the essay heading the simulation belongs to
    heading_tr: str                      # the same heading in the Turkish essay
    options: tuple[str, ...]             # which of q, r, gap, seed, sigma_r, sigma_theta the simulation reads


@dataclass(frozen=True)
class Lesson:
    number: int
    name: tuple[str, str]
    essay: tuple[str, str]               # (English, Turkish) URL of the lesson's essay
    entries: tuple[Entry, ...]           # empty: still being written


LESSONS = (
    Lesson(1, ("Kalman filter", "Kalman filtresi"),
           (ESSAYS + "fusion-01-kalman-filter", ESSAYS + "fusion-01-kalman-filter-tr"), (
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
    Lesson(2, ("EKF", "EKF"),
           (ESSAYS + "fusion-02-extended-kalman-filter", ESSAYS + "fusion-02-extended-kalman-filter-tr"), (
        Entry("ekf-02-tangent", ("a belief through a curve, and its tangent", "bir inanç eğriden geçiyor, yanında teğeti"),
              "2. The tangent line", "2. Teğet doğrusu", ()),
        Entry("ekf-03-range-only", ("one EKF update with the range alone", "yalnız menzille bir EKF güncellemesi"),
              "3. One update by hand", "3. Elle bir güncelleme", ("sigma_r",)),
        Entry("ekf-05-radar-update", ("one update with range and bearing", "menzil ve kerterizle bir güncelleme"),
              "5. One radar update by hand", "5. Elle bir radar güncellemesi", ("sigma_r", "sigma_theta")),
        Entry("ekf-07-flyby", ("tracking a fly-by with the EKF", "bir geçişi EKF ile takip etmek"),
              "7. Python from scratch", "7. Python ile sıfırdan", ("q", "sigma_r", "sigma_theta", "seed")),
        Entry("ekf-08-wrap", ("the bearing wraps: with the fix and without", "kerteriz başa sarıyor: düzeltmeli ve düzeltmesiz"),
              "8. The angle wraps", "8. Açı başa sarınca", ("q", "sigma_r", "sigma_theta", "seed")),
        Entry("ekf-09-consistency", ("where the tangent fails: four cases", "teğet nerede yetmez: dört durum"),
              "9. Where the tangent fails", "9. Teğet nerede yetmez?", ("seed",)),
    )),
    Lesson(3, ("UKF", "UKF"),
           (ESSAYS + "fusion-03-unscented-kalman-filter", ESSAYS + "fusion-03-unscented-kalman-filter-tr"), (
        Entry("ukf-02-icd-decode", ("decode one LOS message of the example ICD", "örnek ICD'nin bir LOS mesajını çöz"),
              "2. Reading an ICD", "2. Bir ICD okumak", ()),
        Entry("ukf-03-square", ("a belief through y = x²: tangent, three points, truth",
                                "bir inanç y = x²'den geçiyor: teğet, üç nokta, gerçek"),
              "3. A belief through a curve", "3. Bir inanç eğriden geçince", ()),
        Entry("ukf-05-sigma-points", ("sigma points in one and two dimensions", "bir ve iki boyutta sigma noktaları"),
              "5. Sigma points", "5. Sigma noktaları", ()),
        Entry("ukf-07-radar-update", ("one UKF update with the range alone", "yalnız menzille bir UKF güncellemesi"),
              "7. One radar update by hand", "7. Elle bir radar güncellemesi", ("sigma_r",)),
        Entry("ukf-08-wide-belief", ("a wide belief: EKF and UKF part ways", "geniş bir inanç: EKF ile UKF ayrışıyor"),
              "8. When the belief is wide", "8. İnanç genişken", ("sigma_r",)),
    )),
)


def entries() -> list[Entry]:
    return [e for lesson in LESSONS for e in lesson.entries]


def entry(sid: str) -> Entry:
    return next(e for e in entries() if e.id == sid)


def lesson_of(sid: str) -> Lesson:
    """The lesson simulation `sid` belongs to."""
    return next(lesson for lesson in LESSONS if any(e.id == sid for e in lesson.entries))


def anchor(heading: str) -> str:
    """The id the site gives a `##` heading: lowercase, only [a-z0-9_ -], spaces to '-', '-' runs collapsed.

    As on the site, letters outside a-z are dropped, so Turkish ı ş ğ ü ö ç vanish ("İ" lowercases to "i" plus a
    combining dot, and only the dot is dropped).
    """
    kept = re.sub(r"[^a-z0-9_ -]", "", heading.lower())
    return re.sub(r"-+", "-", kept.replace(" ", "-"))
