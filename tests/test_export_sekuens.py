"""Tes ekspor sekuens produk (src/export_sekuens.py) dengan data buatan kecil; tanpa model dan jaringan."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src import export_sekuens as es


def _row(rid, pid, ts="2026-01-01 10:00:00", row_order=0, status="ok", kosong=False, include=True,
         p=(0.1, 0.2, 0.7), review="bagus", has_time=True, dup=False, toko="Toko", produk=None):
    return {
        "review_id": rid, "product_id": pid, "nama_toko": toko,
        "nama_produk": f"HP {pid}" if produk is None else produk,
        "review": review, "timestamp": ts, "timestamp_has_time": has_time, "timestamp_status": status,
        "row_order": row_order, "include_for_inference": include, "text_proposal_kosong": kosong,
        "is_text_duplicate": dup, "p_negatif": p[0], "p_netral": p[1], "p_positif": p[2],
        "label_argmax": "positif",
    }


def _df(rows):
    return pd.DataFrame(rows)


def _to_input(df: pd.DataFrame) -> pd.DataFrame:
    """Simulasikan pembacaan CSV (semua kolom teks)."""
    return es.read_input_bytes(es.to_csv_bytes(df))


def _audit(rows, min_ulasan=3):
    return es.bangun_audit(es.ganti_product_id(_to_input(_df(rows))), min_ulasan)


def test_prioritas_alasan():
    rows = [
        _row("r1", "P", kosong=True, include=False, status="relatif", ts=""),
        _row("r2", "P", include=False, status="relatif", ts=""),
        _row("r3", "P", status="relatif", ts=""),
        _row("r4", "P", status="ok", ts=""),
        _row("r5", "P", status="ok", ts="bukan tanggal"),
        _row("r6", "Q"),
    ]
    alasan = dict(zip(*[_audit(rows)[c] for c in ("review_id", "alasan_keluar")]))
    assert alasan == {
        "r1": es.ALASAN_TEKS_KOSONG,
        "r2": es.ALASAN_BUKAN_INCLUDE,
        "r3": es.ALASAN_TIMESTAMP,
        "r4": es.ALASAN_TIMESTAMP,
        "r5": es.ALASAN_TIMESTAMP,
        "r6": es.ALASAN_PRODUK_KECIL,
    }


def test_ambang_dihitung_setelah_penyaringan():
    # P punya 4 baris, tetapi hanya 2 yang lolos alasan 1-3 -> dikeluarkan sebagai produk kecil.
    rows = [_row("a", "P"), _row("b", "P"), _row("c", "P", kosong=True), _row("d", "P", include=False)]
    audit = _audit(rows)
    assert audit.set_index("review_id")["alasan_keluar"].to_dict() == {
        "a": es.ALASAN_PRODUK_KECIL, "b": es.ALASAN_PRODUK_KECIL,
        "c": es.ALASAN_TEKS_KOSONG, "d": es.ALASAN_BUKAN_INCLUDE,
    }


def test_produk_dua_ulasan_keluar_tiga_masuk():
    rows = [_row("x1", "DUA"), _row("x2", "DUA")] + [_row(f"y{i}", "TIGA", row_order=i) for i in range(3)]
    audit = _audit(rows)
    masuk = audit.set_index("review_id")["masuk_sekuens"]
    assert (masuk[["x1", "x2"]] == "False").all()
    assert (masuk[["y0", "y1", "y2"]] == "True").all()
    a = es.bangun_sekuens(audit)
    assert set(a["product_id"]) == {"Toko_HP TIGA"}


def test_relative_estimates_enter_sequences_with_source_file_tiebreak():
    rows = [dict(_row(f"r{i}", "P", status="relatif_dikonversi", has_time=False,
                      ts="2026-07-29", row_order=i), source_file=file)
            for i, file in enumerate(["b.csv", "a.csv", "a.csv"])]
    audit = _audit(rows)
    assert audit.masuk_sekuens.eq("True").all()
    assert audit.sort_values("seq_sekuens").review_id.tolist() == ["r1", "r2", "r0"]
    shuffled = _audit([rows[2], rows[0], rows[1]])
    pd.testing.assert_series_equal(
        audit.set_index("review_id").seq_sekuens.sort_index(),
        shuffled.set_index("review_id").seq_sekuens.sort_index(),
    )


def test_pemecah_seri_timestamp_row_order_review_id():
    rows = [
        _row("z", "P", ts="2026-01-02 00:00:00", row_order=0),
        _row("c", "P", ts="2026-01-01 10:00:00", row_order=5),
        _row("b", "P", ts="2026-01-01 10:00:00", row_order=1),
        _row("a", "P", ts="2026-01-01 10:00:00", row_order=1),
        _row("d", "P", ts="2026-01-01", row_order=9),  # tanggal saja = 00:00:00
        _row("e", "P", ts="2026-01-01 10:00:00", row_order=10),  # row_order numerik, bukan string
    ]
    audit = _audit(rows)
    urut = audit.assign(_s=audit["seq_sekuens"].astype(int)).sort_values("_s")
    assert urut["review_id"].tolist() == ["d", "a", "b", "c", "e", "z"]
    a = es.bangun_sekuens(audit)
    assert a["seq"].tolist() == ["1", "2", "3", "4", "5", "6"]
    assert a["timestamp"].tolist() == urut["timestamp"].tolist()


def test_seq_berurutan_per_produk_dan_urutan_B_sama_input():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        pid = f"P{i % 4}"
        rows.append(_row(f"r{i:02d}", pid, ts=f"2026-01-{1 + int(rng.integers(0, 5)):02d} 10:00:00",
                         row_order=int(rng.integers(0, 3)), kosong=bool(i % 11 == 0)))
    df = _to_input(_df(rows))
    audit = es.bangun_audit(df)
    assert audit["review_id"].tolist() == df["review_id"].tolist()
    a = es.bangun_sekuens(audit)
    for _, g in a.groupby("product_id"):
        assert g["seq"].astype(int).tolist() == list(range(1, len(g) + 1))
        assert pd.to_datetime(g["timestamp"]).is_monotonic_increasing


def test_A_bisa_direkonstruksi_dari_B():
    rows = [_row(f"r{i}", f"P{i % 3}", ts=f"2026-02-{1 + i % 7:02d} 08:00:00", row_order=i,
                 p=(0.2, 0.3, 0.5), include=i != 4) for i in range(15)]
    berkas = es.hasilkan_berkas(es.to_csv_bytes(_df(rows)))
    b = es.read_input_bytes(berkas["audit"])
    a = es.read_input_bytes(berkas["sekuens"])
    rekon = b[b["masuk_sekuens"] == "True"].copy()
    rekon["seq"] = rekon["seq_sekuens"]
    rekon = rekon.sort_values(["product_id", "seq"], key=lambda s: s.astype(int) if s.name == "seq" else s)
    pd.testing.assert_frame_equal(rekon[es.SEKUENS_COLUMNS].reset_index(drop=True), a)


def test_kolom_probabilitas_dipilih_menurut_nama():
    rows = [_row(f"r{i}", "P", row_order=i, p=(0.11, 0.22, 0.67)) for i in range(3)]
    df = _df(rows)
    cols = list(df.columns)
    rng = np.random.default_rng(42)
    acak = [cols[i] for i in rng.permutation(len(cols))]
    berkas = es.hasilkan_berkas(es.to_csv_bytes(df[acak]))
    a = es.read_input_bytes(berkas["sekuens"])
    b = es.read_input_bytes(berkas["audit"])
    assert list(a.columns) == es.SEKUENS_COLUMNS
    assert (a["p_positif"] == "0.67").all()
    assert (a["p_netral"] == "0.22").all()
    assert (a["p_negatif"] == "0.11").all()
    pos = acak.index("product_id")
    assert list(b.columns) == acak[: pos + 1] + [es.PRODUCT_ID_LAMA] + acak[pos + 1:] + es.AUDIT_EXTRA_COLUMNS
    inv = es.periksa_invarian(es.read_input_bytes(es.to_csv_bytes(df[acak])), berkas)
    assert all(x["lolos"] for x in inv), inv


def test_teks_dengan_koma_kutip_dan_u2028():
    teks = ['bagus, "mantap" sekali', "baris pemisah unicode", 'kutip "" ganda\nbaris baru', "biasa"]
    rows = [_row(f"r{i}", "P", row_order=i, review=t) for i, t in enumerate(teks)]
    df = _df(rows)
    berkas = es.hasilkan_berkas(es.to_csv_bytes(df))
    b = es.read_input_bytes(berkas["audit"])
    assert len(b) == 4
    assert b["review"].tolist() == teks
    assert berkas["audit"].startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in berkas["audit"]
    inv = es.periksa_invarian(es.read_input_bytes(es.to_csv_bytes(df)), berkas)
    assert all(x["lolos"] for x in inv), inv


def test_invarian_mendeteksi_A_rusak():
    rows = [_row(f"r{i}", "P", row_order=i) for i in range(3)]
    df = _df(rows)
    berkas = es.hasilkan_berkas(es.to_csv_bytes(df))
    a = es.read_input_bytes(berkas["sekuens"])
    a.loc[0, "p_positif"] = "0.9"
    rusak = dict(berkas, sekuens=es.to_csv_bytes(a))
    hasil = {x["invarian"]: x["lolos"] for x in es.periksa_invarian(_to_input(df), rusak)}
    assert not hasil["A_sama_B_masuk"]
    assert not hasil["p_jumlah_1"]


def test_kolom_wajib_hilang_ditolak():
    df = _df([_row("r1", "P")]).drop(columns=["p_netral"])
    with pytest.raises(es.EksporError, match="p_netral"):
        es.hasilkan_berkas(es.to_csv_bytes(df))


def _cli_args(tmp_path):
    return [
        "--input", str(tmp_path / "in.csv"),
        "--out-sekuens", str(tmp_path / "sentiment" / "sekuens_produk.csv"),
        "--out-audit", str(tmp_path / "audit" / "sekuens_audit.csv"),
        "--min-ulasan", "3",
    ]


def _tulis_input(tmp_path):
    rows = [_row(f"r{i}", f"P{i % 3}", row_order=i, kosong=i == 7) for i in range(12)]
    (tmp_path / "in.csv").write_bytes(es.to_csv_bytes(_df(rows)))


def test_cli_end_to_end_dan_input_tidak_berubah(tmp_path):
    _tulis_input(tmp_path)
    before = hashlib.sha256((tmp_path / "in.csv").read_bytes()).hexdigest()
    assert es.main(_cli_args(tmp_path)) == es.EXIT_OK
    assert hashlib.sha256((tmp_path / "in.csv").read_bytes()).hexdigest() == before
    audit_dir = tmp_path / "audit"
    for name in ("sekuens_audit.csv", es.NAMA_DIKELUARKAN, es.NAMA_LAPORAN, es.NAMA_RINGKASAN, es.NAMA_CHECKSUMS):
        assert (audit_dir / name).is_file(), name
    d = es.read_input_bytes((audit_dir / es.NAMA_DIKELUARKAN).read_bytes())
    assert list(d.columns) == es.DIKELUARKAN_COLUMNS
    assert d["alasan_keluar"].tolist() == [es.ALASAN_TEKS_KOSONG]


def test_cli_menolak_menimpa(tmp_path):
    _tulis_input(tmp_path)
    target = tmp_path / "sentiment" / "sekuens_produk.csv"
    target.parent.mkdir()
    target.write_text("lama", encoding="utf-8")
    assert es.main(_cli_args(tmp_path)) == es.EXIT_STOPPED
    assert target.read_text(encoding="utf-8") == "lama"
    assert not (tmp_path / "audit").exists()


def test_reproduksibel(tmp_path):
    _tulis_input(tmp_path)
    data = (tmp_path / "in.csv").read_bytes()
    assert es.hasilkan_berkas(data) == es.hasilkan_berkas(data)


# ------------------------------------------------------------
# product_id baru = nama_toko + "_" + nama_produk
# ------------------------------------------------------------

def test_bentuk_product_id_huruf_asli():
    df = _df([
        _row("r1", "1688.shopp|oppo a31", toko="1688.shopp", produk="Oppo A31"),
        _row("r2", "x|y", toko="  Toko Besar ", produk=" Vivo  Y20 RAM 8+256GB  "),
    ]).astype(str)
    assert es.bentuk_product_id(df).tolist() == ["1688.shopp_Oppo A31", "Toko Besar_Vivo  Y20 RAM 8+256GB"]
    b = es.ganti_product_id(df)
    assert list(b.columns[:3]) == ["review_id", "product_id", es.PRODUCT_ID_LAMA]
    assert b[es.PRODUCT_ID_LAMA].tolist() == ["1688.shopp|oppo a31", "x|y"]


def test_konflik_satu_id_lama_dua_ejaan():
    rows = [_row(f"r{i}", "toko|hp a", toko="Toko", produk="HP A", row_order=i) for i in range(3)]
    rows.append(_row("r9", "toko|hp a", toko="Toko", produk="Hp A"))
    with pytest.raises(es.KonflikPemetaanError) as exc:
        es.hasilkan_berkas(es.to_csv_bytes(_df(rows)))
    varian = exc.value.konflik["lama_banyak_ejaan"]
    assert {(v["nama_produk"], v["baris"]) for v in varian} == {("HP A", 3), ("Hp A", 1)}
    assert "'Hp A') 1 baris" in str(exc.value)


def test_konflik_dua_id_lama_satu_id_baru():
    # Garis bawah di nama membuat dua produk berbeda bertabrakan: "a_b" + "c" dan "a" + "b_c" -> "a_b_c".
    rows = [_row(f"x{i}", "a_b|c", toko="a_b", produk="c", row_order=i) for i in range(3)]
    rows += [_row(f"y{i}", "a|b_c", toko="a", produk="b_c", row_order=i) for i in range(2)]
    with pytest.raises(es.KonflikPemetaanError) as exc:
        es.hasilkan_berkas(es.to_csv_bytes(_df(rows)))
    tabrakan = exc.value.konflik["baru_dari_banyak_lama"]
    assert {(t["baru"], t["lama"], t["baris"]) for t in tabrakan} == {("a_b_c", "a_b|c", 3), ("a_b_c", "a|b_c", 2)}


def test_nama_kosong_ditolak():
    rows = [_row(f"r{i}", "p", toko="", produk="HP", row_order=i) for i in range(3)]
    with pytest.raises(es.KonflikPemetaanError, match="nama_toko kosong: 3"):
        es.hasilkan_berkas(es.to_csv_bytes(_df(rows)))


def test_cli_berhenti_pada_konflik_tanpa_menulis(tmp_path, capsys):
    rows = [_row("r1", "p|1", toko="T", produk="A"), _row("r2", "p|1", toko="T", produk="a")]
    (tmp_path / "in.csv").write_bytes(es.to_csv_bytes(_df(rows)))
    assert es.main(_cli_args(tmp_path)) == es.EXIT_STOPPED
    assert "lebih dari satu ejaan" in capsys.readouterr().out
    assert not (tmp_path / "audit").exists() and not (tmp_path / "sentiment").exists()


def test_A_tanpa_review_id_B_dengan_review_id_kunci_unik():
    rows = [_row(f"r{i}", f"P{i % 2}", ts=f"2026-03-0{1 + i % 3} 09:00:00", row_order=i % 2) for i in range(10)]
    berkas = es.hasilkan_berkas(es.to_csv_bytes(_df(rows)))
    a = es.read_input_bytes(berkas["sekuens"])
    b = es.read_input_bytes(berkas["audit"])
    assert "review_id" not in a.columns and list(a.columns) == es.SEKUENS_COLUMNS
    assert "review_id" in b.columns and b["review_id"].is_unique
    assert not a.duplicated(["product_id", "seq"]).any()
    b_masuk = b[b["masuk_sekuens"] == "True"]
    assert not b_masuk.duplicated(["product_id", "seq_sekuens"]).any()
    gabung = a.merge(b_masuk, left_on=["product_id", "seq"], right_on=["product_id", "seq_sekuens"],
                     suffixes=("", "_b"), validate="one_to_one")
    assert len(gabung) == len(a) == len(b_masuk)
    for c in es.PROB_COLUMNS_SEKUENS:
        assert (gabung[c] == gabung[f"{c}_b"]).all()


def test_nama_dengan_koma_kutip_spasi_ganda_garis_bawah():
    nama = [("Toko, Jaya", 'HP "Pro" 5G'), ("Toko  Dua", "Redmi_Note 9"), ("toko_ku", "Samsung, A12  ")]
    rows = []
    for j, (toko, produk) in enumerate(nama):
        rows += [_row(f"r{j}{i}", f"old{j}", toko=toko, produk=produk, row_order=i) for i in range(3)]
    df = _df(rows)
    berkas = es.hasilkan_berkas(es.to_csv_bytes(df))
    a = es.read_input_bytes(berkas["sekuens"])
    assert sorted(set(a["product_id"])) == sorted(
        ['Toko, Jaya_HP "Pro" 5G', "Toko  Dua_Redmi_Note 9", "toko_ku_Samsung, A12"])
    inv = es.periksa_invarian(_to_input(df), berkas)
    assert all(x["lolos"] for x in inv), [x for x in inv if not x["lolos"]]


def test_seq_sama_dengan_pengelompokan_lama():
    rows = [_row(f"r{i:02d}", f"old{i % 3}", toko="T", produk=f"Prod {i % 3}",
                 ts=f"2026-01-{1 + (i * 7) % 9:02d} 10:00:00", row_order=i % 4) for i in range(30)]
    df = _to_input(_df(rows))
    baru = es.bangun_audit(es.ganti_product_id(df))
    lama = es.bangun_audit(df)
    assert baru["seq_sekuens"].equals(lama["seq_sekuens"])
