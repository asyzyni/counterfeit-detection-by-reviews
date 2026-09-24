# Metodologi: Deteksi Suspicion of Counterfeit dari Ulasan Pengguna

## 1\. Diagram Alur Pipeline \(Arsitektur Final / v3\)



**Titik kritis anti-leakage** (ditandai garis putus-putus di diagram):

* Split (L) terjadi **sebelum** HMM di-fit (O), dan hanya produk **train** yang dipakai untuk fitting.
* HMM tidak pernah melihat `suspicion_score` — parameter dibekukan lalu diterapkan ke train & test.

- - -

<br>
<br>
