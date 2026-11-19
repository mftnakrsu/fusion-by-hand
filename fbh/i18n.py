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
    "deriv_position_noise": ("position noise", "konum gürültüsü"),
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
    "nis_over_detail": ("filter R = {r}, sensor R = {r_sensor}", "filtre R = {r}, sensör R = {r_sensor}"),
    "nis_qsmall": ("Q too small", "Q çok küçük"),
    "nis_inside": ("around the maneuver {inside}, elsewhere {outside}", "manevra çevresinde {inside}, başka yerde {outside}"),
    "nis_expected": ("expected mean: 1", "beklenen ortalama: 1"),
    "nis_bound": ("χ² 95 % bound", "χ² %95 sınırı"),
    "nis_window": ("average over 20 frames", "20 karelik ortalama"),
    "nis_single": ("single frame", "tek kare"),
    # lesson 2: shared axes and legend entries
    "scan": ("scan", "tarama"),
    "east_m": ("east (m)", "doğu (m)"),
    "north_m": ("north (m)", "kuzey (m)"),
    "range_m": ("range r (m)", "menzil r (m)"),
    "density": ("density", "yoğunluk"),
    "radar": ("radar", "radar"),
    "reports": ("reports", "ölçümler"),
    "true_path": ("true path", "gerçek yol"),
    "ekf_estimate": ("EKF estimate", "EKF kestirimi"),
    "line_of_sight": ("line of sight", "bakış doğrultusu"),
    "exact": ("exact answer", "tam cevap"),
    # ekf-02-tangent
    "tan_title_curve": ("The range curve and its tangents", "Menzil eğrisi ve teğetleri"),
    "tan_title_belief": ("{mean} ± {sigma} m, pushed through the curve", "{mean} ± {sigma} m eğriden geçince"),
    "tan_x": ("position along the flight line pₓ (m)", "uçuş hattı boyunca konum pₓ (m)"),
    "tan_curve": ("range r = √(pₓ² + {offset}²)", "menzil r = √(pₓ² + {offset}²)"),
    "tan_at": ("tangent at pₓ = {x} m, slope {slope}", "teğet: pₓ = {x} m, eğim {slope}"),
    "tan_belief": ("belief {mean} ± {sigma} m (±2σ band)", "inanç {mean} ± {sigma} m (±2σ bandı)"),
    "tan_true": ("what comes out: mean {mean} m, σ {std} m", "gerçekte çıkan: ortalama {mean} m, σ {std} m"),
    "tan_lin": ("what the tangent says: mean {mean} m, σ {std} m", "teğetin dediği: ortalama {mean} m, σ {std} m"),
    # ekf-03-range-only
    "ro_title": ("One update with the range alone: reading {z} m, σᵣ = {sigma_r} m",
                 "Yalnız menzille bir güncelleme: ölçüm {z} m, σᵣ = {sigma_r} m"),
    "ro_x": ("position pₓ (m)", "konum pₓ (m)"),
    "ro_reading": ("the reading alone: √({z}² − {offset}²) = {x}", "yalnız ölçüm: √({z}² − {offset}²) = {x}"),
    # ekf-05-radar-update
    "ru_title": ("One radar update: range and bearing", "Bir radar güncellemesi: menzil ve kerteriz"),
    "ru_prediction": ("prediction {x} and its 2σ circle", "tahmin {x} ve 2σ çemberi"),
    "ru_report": ("report: {r} m at {theta} rad", "ölçüm: {theta} rad kerterizde {r} m"),
    "ru_circle_line": ("its range circle and bearing line", "ölçümün menzil çemberi ve kerteriz doğrusu"),
    "ru_corrected": ("corrected estimate {x} and its 2σ ellipse", "düzeltilmiş kestirim {x} ve 2σ elipsi"),
    "ru_variances": ("variance after the update:\n{along} m² along the line of sight,\n{across} m² across it"
                     "\n(the prediction: {prior} m² in every direction)",
                     "güncellemeden sonra varyans:\nbakış doğrultusu boyunca {along} m²,\nona dik {across} m²"
                     "\n(tahminde her yönde {prior} m²)"),
    # ekf-07-flyby
    "fb_title_map": ("A fly-by, {north} m north of the radar", "Radarın {north} m kuzeyinden bir geçiş"),
    "fb_title_error": ("Distance to the truth, per scan", "Tarama başına gerçeğe uzaklık"),
    "fb_distance": ("distance to the truth (m)", "gerçeğe uzaklık (m)"),
    "fb_raw": ("a report on its own (rms {rms} m)", "tek başına bir ölçüm (rms {rms} m)"),
    "fb_filter": ("EKF estimate (rms {rms} m)", "EKF kestirimi (rms {rms} m)"),
    "fb_band": ("0 to 2σ, the σ the filter claims", "0 ile 2σ arası, filtrenin söylediği σ"),
    # ekf-08-wrap
    "wr_title_map": ("Crossing the line where the bearing wraps", "Kerterizin başa sardığı çizgiyi geçerken"),
    "wr_title_bearing": ("Reported and expected bearing", "Ölçülen ve beklenen kerteriz"),
    "wr_cut": ("bearing ±π: due west of the radar", "kerteriz ±π: radarın tam batısı"),
    "wr_with": ("with the wrap (worst error {error} m)", "sarma ile (en büyük hata {error} m)"),
    "wr_without": ("without the wrap (worst error {error} m)", "sarma olmadan (en büyük hata {error} m)"),
    "wr_leaves": ("leaves the map\nat scan {k}", "{k}. taramada\nharitadan çıkıyor"),
    "wr_bearing": ("bearing (rad)", "kerteriz (rad)"),
    "wr_reported": ("reported", "ölçülen"),
    "wr_expected": ("expected by the filter (with the wrap)", "filtrenin beklediği (sarma ile)"),
    "wr_jump": ("scan {k}:\nplain difference {raw} rad,\nwrapped into [−π, π):\n{wrapped} rad",
                "{k}. tarama:\ndüz fark {raw} rad,\n[−π, π) aralığına\nsarılınca {wrapped} rad"),
    "wr_no_jump": ("no jump in this run: the reading and\nthe prediction never land on opposite\nsides of ±π, so both runs are the same",
                   "bu koşuda sıçrama yok: ölçüm ile tahmin\nhiçbir taramada ±π'nin iki ayrı yanına\ndüşmüyor, iki koşu aynı"),
    # ekf-09-consistency
    "cs_title": ("Where the tangent fails: the real error against the σ the filter claims",
                 "Teğet nerede yetmez: gerçek hata ve filtrenin söylediği σ"),
    "cs_baseline": ("baseline", "temel durum"),
    "cs_vague_prior": ("vague prior", "belirsiz ilk tahmin"),
    "cs_first_report": ("start from the first report", "ilk ölçümden başlat"),
    "cs_sharp_range": ("sharp range, coarse bearing", "keskin menzil, kaba kerteriz"),
    "cs_panel": ("{name}: bend {bend} m, σᵣ {sigma_r} m", "{name}: bükülme {bend} m, σᵣ {sigma_r} m"),
    "cs_rms": ("rms position error over {runs} runs", "{runs} denemede rms konum hatası"),
    "cs_claimed": ("the σ the filter claims", "filtrenin söylediği σ"),
    "cs_error": ("position error (m)", "konum hatası (m)"),
}

LANGS = ("en", "tr")
KEYS = tuple(_TEXT)


def t(key: str, lang: str) -> str:
    """The text for `key` in `lang` ("en" or "tr"); an unknown key or language is a KeyError."""
    if lang not in LANGS:
        raise KeyError(f"unknown language {lang!r}, expected one of {LANGS}")
    return _TEXT[key][LANGS.index(lang)]
