
📌 Makine Öğrenmesinde Matris Manipülasyonu, Özdeğerler ve Özvektörler

🔹 1. Tanımlar

- Matris: Sayılardan oluşan iki boyutlu dizilerdir. Veri yapılarının, dönüşümlerin ve lineer denklemlerin ifade edilmesinde kullanılır.
- Özdeğer (Eigenvalue): Bir A matrisine uygulanan dönüşümde, belirli bir vektörün (özvektör) yalnızca skala çarpanı kadar değişmesini sağlayan sabit sayılardır.
- Özvektör (Eigenvector): Bir matrisle çarpıldığında yönü değişmeyen (yalnızca büyüklüğü değişen) vektörlerdir.

🔹 2. Matrislerin Makine Öğrenmesindeki Rolü

Makine öğrenmesi modellerinde matrisler veri temsili, parametre güncelleme, dönüşüm ve boyut indirgeme gibi birçok aşamada aktif olarak kullanılır. Özellikle doğrusal cebir, temel taşlardan biridir.

🔹 3. Özdeğer ve Özvektörlerin Kullanım Alanları

- PCA (Principal Component Analysis): Boyut indirgeme tekniğidir. Verideki en fazla varyansı taşıyan yönler özvektörler ile bulunur.
- Spektral Kümeleme: Özdeğer ayrıştırması, veri noktaları arasındaki ilişkiyi yansıtan benzerlik matrislerine uygulanır.
- Doğrusal Transformasyonlar: Görsel veri veya özellik dönüşümlerinde matris çarpımları ve özvektör ayrışımları kullanılır.

🔹 4. Kullanılan Kaynaklar

- [Introduction to Matrices for Machine Learning](https://machinelearningmastery.com/introduction-matrices-machine-learning/)
- [Eigendecomposition, Eigenvalues and Eigenvectors](https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/)
- [NumPy `linalg.eig` Documentation](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html)
- [NumPy Source Code for `linalg`](https://github.com/numpy/numpy/tree/main/numpy/linalg)
- [Eigenvalue Calculation Without NumPy](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors)
