"""English and Turkish text for every axis, legend entry and title of the figures."""

_TEXT = {
    # shared axes and legend entries
    "frame": ("frame", "kare"),
    "position_px": ("position (px)", "konum (px)"),
    "pos_error_px": ("position error (px)", "konum hatası (px)"),
    "velocity": ("velocity (px/frame)", "hız (px/kare)"),
    "vel_error": ("velocity error (px/frame)", "hız hatası (px/kare)"),
    "measurement": ("measurement", "ölçüm"),
    "filter": ("filter", "filtre"),
    "truth": ("truth", "gerçek"),
    "band_2s": ("±2σ band", "±2σ bandı"),
    "mean": ("mean", "ortalama"),
    # 01-fusion
    "fusion_title": ("Two sources, one estimate", "İki kaynak, tek tahmin"),
    "fusion_x": ("temperature (°C)", "sıcaklık (°C)"),
    "fusion_density": ("density", "yoğunluk"),
    "fusion_good": ("good thermometer", "kaliteli termometre"),
    "fusion_guess": ("guess by eye", "göz kararı"),
    "fusion_fused": ("fused", "birleşik"),
    "variance": ("variance", "varyans"),
    # 03-1d-loop
    "loop_title_pos": ("Predict, update, then no measurement", "Tahmin, güncelleme, sonra ölçüm yok"),
    "loop_title_var": ("Uncertainty goes up, then down", "Belirsizlik artar, sonra azalır"),
    "prediction": ("prediction", "tahmin"),
    "update": ("update", "güncelleme"),
    "no_measurement": ("no measurement", "ölçüm yok"),
    "variance_px2": ("P (px²)", "P (px²)"),
    # 05-derivative-vs-filter
    "deriv_title": ("A target that does not move", "Hiç hareket etmeyen hedef"),
    "deriv_difference": ("difference of two boxes", "iki kutunun farkı"),
    "deriv_ylabel": ("velocity estimate (px/frame)", "hız tahmini (px/kare)"),
    # 08-covariance
    "cov_before": ("before the prediction: P", "tahminden önce: P"),
    "cov_after": ("after the prediction: F P Fᵀ", "tahminden sonra: F P Fᵀ"),
    "cov_cloud": ("error cloud", "hata bulutu"),
    "cov_ellipse": ("2σ ellipse", "2σ elipsi"),
    "covariance": ("covariance", "kovaryans"),
    # 12-velocity-learning
    "learn_title": ("The camera never measures speed, the filter learns it", "Kamera hızı ölçmez, filtre öğrenir"),
    "learn_start": ("starting belief", "başlangıç inancı"),
    "learn_truth": ("true speed", "gerçek hız"),
    "learn_estimate": ("filter's speed", "filtrenin hızı"),
    # 13-q-r-tradeoff
    "trade_learn": ("learning the speed (truth 13)", "hızı öğrenme (gerçek 13)"),
    "trade_maneuver": ("after a maneuver", "manevradan sonra"),
    "trade_calm": ("calm, slow to follow", "sakin, geç uyar"),
    "trade_agile": ("agile, jittery", "çevik, titrek"),
    "maneuver": ("maneuver", "manevra"),
    # 14-nis
    "nis_title": ("Is the filter honest? NIS over time", "Filtre dürüst mü? Zamana göre NIS"),
    "nis_ylabel": ("NIS", "NIS"),
    "nis_honest": ("honest filter", "dürüst filtre"),
    "nis_over": ("overconfident filter", "fazla emin filtre"),
    "nis_expected": ("expected mean: 1", "beklenen ortalama: 1"),
    "nis_bound": ("χ² 95 % bound", "χ² %95 sınırı"),
    "nis_window": ("average over 20 frames", "20 karelik ortalama"),
}

LANGS = ("en", "tr")
KEYS = tuple(_TEXT)


def t(key: str, lang: str) -> str:
    """The text for `key` in `lang` ("en" or "tr"); an unknown key or language is a KeyError."""
    if lang not in LANGS:
        raise KeyError(f"unknown language {lang!r}, expected one of {LANGS}")
    return _TEXT[key][LANGS.index(lang)]
