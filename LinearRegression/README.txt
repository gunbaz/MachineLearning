Linear Regression - Genel Açıklama

Bu proje, doğrusal regresyon (Linear Regression) yönteminin temelini anlamak ve uygulamak amacıyla hazırlanmıştır. Doğrusal regresyon, bağımlı bir değişken ile bir veya daha fazla bağımsız değişken arasındaki ilişkiyi modelleyen istatistiksel bir yöntemdir.

 Yöntem

Model şu formülle ifade edilir:
    ŷ = Xθ

Burada:
- ŷ: Tahmin edilen değerler
- X: Girdi özellikleri (bağımsız değişkenler)
- θ: Katsayı vektörü (öğrenilen parametreler)

Amaç, hata (maliyet) fonksiyonu olan ortalama kare hatayı (MSE) minimize etmektir:
    MSE = (1/n) * Σ (yᵢ - xᵢᵀθ)²

 Kullanım Alanları
- Ev fiyat tahmini
- Talep tahmini
- İstatistiksel modelleme
- Finansal analiz

 Dosya İçeriği
Bu klasör genellikle aşağıdaki dosyaları içerir:
- `LinearRegressionWLSE.ipynb`: En küçük kareler yöntemiyle regresyon eğitimi.
- `LinearRegressionWSLearn.ipynb`: Scikit-learn kullanarak regresyon eğitimi.
- `data: CSV veri dosyaları.
- `README.md`: Bu belge.


