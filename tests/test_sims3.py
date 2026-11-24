import json
import struct

import numpy as np
import pytest

from fbh import registry, sims3
from fbh.base import Options


def test_icd_table_is_the_whole_message_once():
    rows = sims3.decode()
    covered = []
    for r in rows:
        a, b = (int(x) for x in (r["bytes"].split("-") * 2)[:2])
        covered += range(a, b + 1)
    assert covered == list(range(32))                                       # every byte, in order, exactly once
    assert [r["field"] for r in rows] == [f[0] for f in sims3.FIELDS]


def test_icd_decodes_another_message_by_the_same_rules():
    """A message built with struct from chosen values decodes back to them: the layout, not the example, is coded."""
    values = (0xA55A, 0x10, 1, 24, 7, 123, -(2**31), 2**31 - 1, -32768, 32767, 65535, 2, 0x10)
    msg = struct.pack(">HBBHHQiihhHBB", *values)
    v = {r["field"]: r for r in sims3.decode(msg)}
    assert [r["raw"] for r in v.values()] == list(values)
    assert v["az"]["value"] == pytest.approx(-np.pi) and v["el"]["value"] == pytest.approx(np.pi, rel=1e-9)
    assert v["az_rate"]["value"] == pytest.approx(-3.2768) and v["sensor"]["value"] == "LWIR"
    assert sims3.icd_check(list(v.values()))["gimbal_at_limit"] == 1


def test_ukf_update_follows_the_reading():
    """Readings farther than the prediction expects push the estimate farther out, and the gain stays put."""
    a, b = sims3.range_update(z=506.0), sims3.range_update(z=516.0)
    assert b["x_plus"] > a["x_plus"] and b["K"] == pytest.approx(a["K"]) and b["P_plus"] == pytest.approx(a["P_plus"])


def test_a_sharper_radar_shrinks_the_ukf_variance():
    P = [sims3.range_update(sigma_r=s)["P_plus"] for s in (10.0, 2.0, 0.5)]
    assert P[0] > P[1] > P[2] > 0


def test_wide_belief_ukf_is_honest_where_the_ekf_is_not_for_several_readings():
    for z in (540.0, 560.0, 580.0):
        d = sims3.range_update(P=10_000.0, z=z)
        assert d["ukf_off_sigmas"] < 1 < d["ekf_off_sigmas"], z


@pytest.mark.parametrize("sid", list(sims3.SIMS))
def test_every_lesson_three_id_runs_plots_both_languages_and_is_json_ready(sid, tmp_path):
    res = sims3.SIMS[sid](Options())
    json.dumps(res.table, allow_nan=False), json.dumps(res.summary, allow_nan=False)
    assert res.table and set(res.summary) and all(isinstance(row, dict) for row in res.table)
    assert res.show_table                                                   # worked by hand: the CLI prints the table
    pngs = [res.plot(tmp_path / lang, lang) for lang in ("en", "tr")]
    for png in pngs:
        assert png.name == f"{sid}.png"
        assert struct.unpack(">II", png.read_bytes()[16:24]) == (1600, 900), (sid, png)
    assert pngs[0].read_bytes() != pngs[1].read_bytes()
    assert registry.entry(sid)
