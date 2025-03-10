import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Veri setini yükleme
file_path = 'C://Users//pc//OneDrive//Masaüstü//NaiveBayes//adult_cleaned.csv'  # Veri seti yolunu güncelleyin
data = pd.read_csv(file_path)

# Kategorik değişkenleri sayısal verilere dönüştürme
def encode_data(df, columns):
    encoded_df = df.copy()
    for col in columns:
        encoded_df[col] = encoded_df[col].astype('category').cat.codes
    return encoded_df

# Eksik verileri doldurma
# Eksik verileri doldurma
def fill_missing_data(df):
    # Sürekli sayısal veriler için ortalama ile doldurma
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())  # inplace=True kullanımı kaldırıldı
    # Kategorik veriler için mod ile doldurma
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # inplace=True kullanımı kaldırıldı
    return df

# Veri ön işleme
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
data = encode_data(data, categorical_columns)
data = fill_missing_data(data)

# Özellikler (X) ve hedef (y) değişkenlerini ayırma
X = data.drop(columns=['income'])
y = data['income']

# Eğitim ve test verilerini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model 1: Scikit-learn Gaussian Naive Bayes
start_time = time.time()
gnb_sklearn = GaussianNB()
gnb_sklearn.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time

start_time = time.time()
y_pred_sklearn = gnb_sklearn.predict(X_test)
sklearn_test_time = time.time() - start_time

# Model 2: Sıfırdan yazılmış Gaussian Naive Bayes
class GaussianNBCustom:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}
        self.epsilon = 1e-6  # Küçük epsilon değeri ekleyelim
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + self.epsilon
            self.prior[c] = len(X_c) / len(X)
    
    def predict(self, X):
        predictions = [self._predict_instance(x) for x in X.values]
        return np.array(predictions)
    
    def _predict_instance(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.prior[c])
            likelihood = np.sum(self._safe_log(self._pdf(c, x)))
            posteriors.append(prior + likelihood)
        
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, c, x):
        mean = self.mean[c]
        var = self.var[c]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _safe_log(self, value):
        return np.log(np.maximum(value, self.epsilon))

# Eğitim ve test zamanları (sıfırdan yazılmış model)
start_time = time.time()
gnb_custom = GaussianNBCustom()
gnb_custom.fit(X_train, y_train)
custom_train_time = time.time() - start_time

start_time = time.time()
y_pred_custom = gnb_custom.predict(X_test)
custom_test_time = time.time() - start_time

# Performans ölçümleri
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
cm_custom = confusion_matrix(y_test, y_pred_custom)

# Karmaşıklık matrisi görselleştirme
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title('Scikit-learn Gaussian Naive Bayes - Karmaşıklık Matrisi')

plt.subplot(1, 2, 2)
sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title('Sıfırdan Yazılmış Gaussian Naive Bayes - Karmaşıklık Matrisi')

plt.tight_layout()
plt.show()

# Eğitim ve test sürelerini yazdıralım
print(f"Scikit-learn model eğitim süresi: {sklearn_train_time:.4f} saniye")
print(f"Scikit-learn model test süresi: {sklearn_test_time:.4f} saniye")
print(f"Sıfırdan yazılmış model eğitim süresi: {custom_train_time:.4f} saniye")
print(f"Sıfırdan yazılmış model test süresi: {custom_test_time:.4f} saniye")

# Performans değerlendirmesi
print("Scikit-learn Model Performansı:")
print(classification_report(y_test, y_pred_sklearn))

print("Sıfırdan Yazılmış Model Performansı:")
print(classification_report(y_test, y_pred_custom))
