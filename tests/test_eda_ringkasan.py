"""Fungsi pembantu EDA v4 dengan data sintetis kecil."""

import math

import numpy as np
import pandas as pd
import pytest

from src.eda.ringkasan import (
    PenyimpanGrafik, as_bool, corong_data, entropi, kelas_panjang, kualitas_teks, label_dari_prob,
    matriks_konfusi, pilih_prob, rata2_per_kelas, rata2_per_produk, ringkasan_produk, ringkasan_waktu,
    tabel_terpotong, top_ngram,
)


def reviews():
    return pd.DataFrame({
        "review_id": list("abcdef"), "product_id": ["p1", "p1", "p1", "p2", "p2", "p3"],
        "nama_toko": ["T"] * 6, "nama_produk": ["P1", "P1", "P1", "P2", "P2", "P3"],
        "text_proposal": ["bagus sekali barangnya", "", "ok", "jelek rusak", "jelek rusak", "mantap"],
        "text_proposal_kosong": [False, True, False, False, False, False],
        "include_for_inference": [True, True, True, True, False, True],
        "is_short": [False] * 6, "is_technical_duplicate": [False, False, False, False, True, False],
        "is_text_duplicate": [False, False, False, True, True, False],
        "seller_reply_suspect": [False] * 5 + [True],
        "timestamp": ["2025-01-02 10:00:00", "2025-01-05", None, "2025-02-01", "2025-02-01", ""],
        "timestamp_has_time": [True, False, False, False, False, False],
        "timestamp_status": ["ok", "ok", "kosong", "ok", "ok", "kosong"],
    })


def test_corong_data_tahapan_dan_ambang_produk():
    manifest = pd.DataFrame({"status": ["diproses", "diproses", "dilewati: x", "dikecualikan: y"],
                             "n_baris_sumber": [5, 3, 2, 9]})
    out = corong_data(manifest, reviews(), n_berkas_csv=4, min_ulasan=2).set_index("tahap")["jumlah"]
    assert out["Berkas dipetakan (status ok)"] == 3
    assert out["Berkas diproses (ada kolom ulasan)"] == 2
    assert out["Baris sumber (rekaman CSV)"] == 8
    assert out["include & text_proposal tidak kosong"] == 4   # a, c, d, f
    assert out["... pada produk >= 2 ulasan"] == 2             # hanya p1 (a, c)


def test_ringkasan_produk_dan_kualitas():
    prod = ringkasan_produk(reviews())
    assert prod.iloc[0]["product_id"] == "p1" and prod.iloc[0]["n_ulasan"] == 3
    q = kualitas_teks(reviews()).set_index("indikator")
    assert q.loc["text_proposal kosong", "n"] == 1
    assert q.loc["sangat pendek (1..2 kata)", "n"] == 4   # ok, jelek rusak x2, mantap
    assert q.loc["balasan penjual dicurigai (seller_reply_suspect)", "n"] == 1


def test_kualitas_sangat_pendek_tidak_menghitung_kosong():
    df = reviews()
    q = kualitas_teks(df, short_words=1).set_index("indikator")
    assert q.loc["sangat pendek (1..1 kata)", "n"] == 2   # ok, mantap


def test_waktu_dan_terpotong():
    out, status = ringkasan_waktu(reviews())
    out = out.set_index("kategori")["n"]
    assert out["timestamp valid"] == 4 and out["timestamp dengan jam"] == 1
    assert out["timestamp tidak valid (NaT)"] == 2
    from src.eda.ringkasan import parse_ts, ulasan_per_bulan
    assert ulasan_per_bulan(reviews()).tolist() == [2, 2]
    assert parse_ts(pd.Series(pd.to_datetime(["2025-01-01"]))).notna().all()
    tab = tabel_terpotong(pd.Series([10, 129, 600, np.nan]), [128, 512]).set_index("max_length")
    assert tab.loc[128, "n_terpotong"] == 2 and tab.loc[512, "n_terpotong"] == 1
    assert math.isclose(tab.loc[512, "persen_terpotong"], 33.333)


def test_top_ngram_kata_tugas_dan_urutan_seri():
    top = top_ngram(["barang bagus dan cepat", "bagus yang cepat", "Bagus"], n=1, k=3)
    assert top.iloc[0].tolist() == ["bagus", 3]
    assert "dan" not in set(top["ngram"])
    bi = top_ngram(["barang bagus", "barang bagus sekali"], n=2, k=1)
    assert bi.iloc[0].tolist() == ["barang bagus", 2]


def probs():
    return pd.DataFrame({"p_negatif": [0.1, 1 / 3, np.nan], "p_netral": [0.2, 1 / 3, np.nan],
                         "p_positif": [0.7, 1 / 3, np.nan],
                         "product_id": ["a", "a", "b"], "nama_toko": "T", "nama_produk": "P"})


def test_prob_dipilih_menurut_nama_bukan_posisi():
    df = probs()[["p_positif", "product_id", "p_negatif", "p_netral"]]
    assert pilih_prob(df).columns.tolist() == ["p_negatif", "p_netral", "p_positif"]
    assert label_dari_prob(df).iloc[0] == "positif"
    with pytest.raises(KeyError):
        pilih_prob(df.drop(columns="p_netral"))


def test_entropi_dan_label():
    e = entropi(probs())
    assert math.isclose(e.iloc[1], math.log(3)) and np.isnan(e.iloc[2])
    assert pd.isna(label_dari_prob(probs()).iloc[2])


def test_kelas_panjang_dan_rata2():
    k = kelas_panjang(pd.Series([1, 2, 3, 60]))
    assert k.astype(str).tolist() == ["1-2", "1-2", "3-5", ">50"]
    r = rata2_per_kelas(probs().iloc[:2], kelas_panjang(pd.Series([1, 4])))
    assert r.loc["1-2", "n"] == 1 and math.isclose(r.loc["1-2", "p_positif"], 0.7)
    prod = rata2_per_produk(probs().iloc[:2], min_ulasan=2)
    assert len(prod) == 1 and prod.iloc[0]["n"] == 2


def test_matriks_konfusi_dan_bool(tmp_path):
    tab = matriks_konfusi(pd.Series(["positif", "netral", None]), pd.Series(["positif", "negatif", "netral"]))
    assert tab.loc["positif", "positif"] == 1 and tab.loc["netral", "negatif"] == 1 and tab.values.sum() == 2
    assert as_bool(pd.Series(["True", "false"])).tolist() == [True, False]
    with pytest.raises(ValueError):
        as_bool(pd.Series(["ya"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    saver = PenyimpanGrafik(tmp_path / "fig")
    fig, _ = plt.subplots()
    assert saver.simpan(fig, "uji").name == "01_uji.png" and saver.simpan(fig, "dua").name == "02_dua.png"
    plt.close(fig)
