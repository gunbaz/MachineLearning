
# Naive Bayes İkili Sınıflandırma Uygulaması

## Proje Tanımı

Bu projede, Naive Bayes yöntemi kullanılarak ikili sınıflandırma gerçekleştirilmiştir. İki farklı yaklaşım uygulanmıştır:
- **Scikit-learn Modeli**: `GaussianNB` kullanılarak model eğitimi yapılmıştır.
- **Custom Model**: Python ile sıfırdan oluşturulmuş (custom) Gaussian Naive Bayes modeli.

Amaç, her iki modelin eğitim ve tahmin süreçlerini karşılaştırmak ve performanslarını (confusion matrix ve işlem süreleri) analiz etmektir.

---

## Veri Seti

Proje kapsamında kullanılan veri seti **adult_cleaned.csv** dosyasıdır. Veri setinin özellikleri:

- **Örnek Sayısı**: 32,561 örnek (satır)
- **Özellik Sayısı**: 14 özellik (sütun)
- **Hedef Değişken**: `income` sütunu (<=50K, >50K)

### Veri Ön İşleme:

- Gereksiz sütunlar kaldırılmıştır (örneğin, `fnlwgt` gibi).
- Kategorik sütunlar sayısallaştırılmıştır (örneğin, `income` sütunu `<=50K → 0, >50K → 1` olarak değiştirilmiştir).

---

## Kullanılan Yöntem ve Kütüphaneler

### Kullanılan Kütüphaneler:
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **time** (eğitim ve tahmin sürelerini ölçmek için)

### Yöntem:

1. **Veri Ön İşleme:**
   - Veri seti okunmuş ve gereksiz sütunlar kaldırılmıştır.
   - `income` sütunu sayısallaştırılmıştır.
   - Eğitim (%70) ve test (%30) setleri oluşturulmuştur.

2. **Model Eğitimi:**
   - **Scikit-learn Modeli**: `GaussianNB` kullanılarak model eğitilmiş ve test edilmiştir.
   - **Custom Model**: Python ile sıfırdan oluşturulan `CustomGaussianNB` sınıfı, her sınıf için ortalama, varyans ve öncelik değerlerini hesaplayarak model oluşturmuştur.

3. **Performans Ölçümü:**
   - Her iki model için eğitim (fit) ve tahmin (predict) süreleri `time` modülü kullanılarak ölçülmüştür.
   - Modellerin performansı, confusion matrix kullanılarak analiz edilip görselleştirilmiştir.

---

## Sonuçlar ve Tartışma

- **Performans Karşılaştırması:**
  Her iki modelin oluşturduğu confusion matrix'ler karşılaştırılmıştır. Bu karşılaştırma, modellerin sınıflandırma başarısını gözler önüne sermektedir.

- **İşlem Süreleri:**
  Eğitim ve tahmin süreleri ölçülmüş, böylece modellerin hesaplama verimliliği incelenmiştir.

- **Değerlendirme:**
  Model performansının değerlendirilmesinde veri setinin sınıf dağılımı, özelliklerin yapısı ve yapılan veri ön işleme adımları önemli rol oynamaktadır. Her iki yaklaşım benzer doğruluk sonuçları verebilir; ancak custom modelin geliştirilmesi, temel algoritmanın anlaşılmasına katkı sağlamaktadır.
