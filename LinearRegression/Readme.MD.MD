Bu çalışmada, bir konut veri seti üzerinde iki farklı linear regresyon modeli eğitilmiş ve karşılaştırılmıştır:

1. LinearRegressionWLSE.ipynb

- Yöntem: Least Squares Estimation (Kapalı Form Çözümü)
- Kütüphaneler: NumPy, Pandas, Matplotlib
- Özellikler: total_rooms, median_income
- Hedef: median_house_value
- Formül: θ = (X^T X)^(-1) X^T y

2. LinearRegressionWSLearn.ipynb

- Yöntem: scikit-learn kütüphanesi ile LinearRegression()
- Kütüphaneler: scikit-learn, Pandas, Matplotlib
- Özellikler ve hedef aynı

MSE (Ortalama Kare Hatası) Karşılaştırması

|        Model               |     MSE (Hata)         |
-------------------------------------------------------
Least Squares (kapalı form)  | 6,893,177,011.52       |
scikit-learn LinearRegression| 6,534,894,664.41       |

Yorum:
Scikit-learn modeli biraz daha düşük hata vermiştir. Bunun nedeni, scikit-learn’un bazı regularizasyon ve sayısal kararlılık avantajlarına sahip olabilmesidir. Ancak fark çok büyük değildir.

Katsayı (θ) Karşılaştırması
---------------------------------------------------
| Özellik         | Least Squares  | scikit-learn |
|-----------------|----------------|--------------|
| Bias (θ₀)       | 51468.79       | 53017.80     |
| total_rooms     | 1.6909         | 1.3439       |
| median_income   | 38771.42       | 38904.64     |



Sonuç
- Her iki model de doğrusal ilişkileri başarılı şekilde modellemiştir.
- scikit-learn biraz daha düşük maliyet (MSE) ile daha stabil sonuçlar sunmuştur.
- Ancak küçük veri setlerinde kapalı form çözüm de yeterli doğruluk sağlamaktadır.
