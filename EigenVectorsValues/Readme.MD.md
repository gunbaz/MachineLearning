
📘 YZM212 - 3. Laboratuvar Değerlendirmesi Raporu

 👤 Öğrenci Bilgileri
- Ad Soyad: [Furkan Günbaz]
- Numara: [23291408]
- Ders: Makine Öğrenmesi (YZM212)
- Tarih: 21.04.2025

---

📌 Konu
Bu laboratuvar çalışması, makine öğrenmesi bağlamında matris manipülasyonu, özdeğer ve özvektör hesaplamalarını kapsamaktadır. Çalışma üç temel bölümden oluşmaktadır:

1. Teorik açıklamalar ve kaynak taraması
2. NumPy kütüphanesi ile `eig()` fonksiyonunun kullanımı
3. Özdeğerlerin hazır fonksiyonlar olmadan sembolik yöntemle hesaplanması

---

 🔍 1. Teorik Arka Plan

🔹 Matris, Özdeğer ve Özvektör Nedir?

- Matris: İki boyutlu veri yapısıdır, genellikle verileri veya ilişkileri temsil eder.
- Özdeğer: ( A.v = lambda.v ) denklemini sağlayan skalar (lambda) değeridir.
- Özvektör: Matris ile çarpıldığında yönü değişmeyen vektörlerdir.

🔹 Makine Öğrenmesinde Kullanım Alanları

- PCA (Principal Component Analysis)
- Spektral Kümeleme
- Lineer Transformasyonlar
- Kovaryans Matrislerinin Analizi

🔹 Kullanılan Kaynaklar

- https://machinelearningmastery.com/introduction-matrices-machine-learning/
- https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/
- https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
- https://github.com/numpy/numpy/tree/main/numpy/linalg
- https://github.com/LucasBN/Eigenvalues-and-Eigenvectors

---

🧪 2. NumPy `eig()` Fonksiyonu Kullanımı

Aşağıdaki 3x3 kare matris için NumPy kullanılarak özdeğerler ve özvektörler hesaplanmıştır:


A = 
[
4 -2  1 
0  3 -1 
0  0  2 
]

➕ Kod Parçası

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
```

### ✅ Sonuçlar
- Özdeğerler: [4.0, 3.0, 2.0]
- Özvektörler: Her bir özdeğer için vektörler konsola yazdırılmıştır.

---

🧠 3. Hazır Fonksiyon Kullanmadan Özdeğer Hesaplama

Karakteristik polinom üzerinden çözüm yapılmıştır:

\[
\det(A - \lambda I) = -\lambda^3 + 9\lambda^2 - 26\lambda + 24
\]

🔹 Elde Edilen Özdeğerler
- [2.0, 3.0, 4.0]

Sonuçlar NumPy ile elde edilen değerlerle birebir örtüşmektedir.

---

