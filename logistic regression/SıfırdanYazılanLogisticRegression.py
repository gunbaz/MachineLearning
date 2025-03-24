import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Veri setini yükleyelim
file_path = 'C:\\Users\\pc\\OneDrive\\Masaüstü\\logistic regression\\adult_cleaned.csv'  # Veri seti yolunu güncelleyin
data = pd.read_csv(file_path)

# Kategorik değişkenleri sayısal verilere dönüştürme
def encode_data(df, columns):
    encoded_df = df.copy()
    for col in columns:
        encoded_df[col] = encoded_df[col].astype('category').cat.codes
    return encoded_df

# Eksik verileri doldurma
def fill_missing_data(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Veri ön işleme
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
data = encode_data(data, categorical_columns)
data = fill_missing_data(data)

# Özellikler (X) ve hedef (y) değişkenlerini ayıralım
X = data.drop(columns=['income'])
y = data['income']

# Eğitim ve test verilerini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veri ölçekleme (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Eğitim verisini ölçeklendir
X_test_scaled = scaler.transform(X_test)        # Test verisini aynı şekilde ölçeklendir

# Model: Logistic Regression
start_time = time.time()
logreg = LogisticRegression(max_iter=5000, solver='liblinear')  # max_iter'ı artırdık ve 'liblinear' solver'ı kullandık
logreg.fit(X_train_scaled, y_train)
logreg_train_time = time.time() - start_time

start_time = time.time()
y_pred_logreg = logreg.predict(X_test_scaled)
logreg_test_time = time.time() - start_time

# Performans ölçümleri
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

# Confusion matrix'i görselleştirelim
plt.figure(figsize=(7, 5))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Eğitim ve test sürelerini yazdıralım
print(f"Logistic Regression model eğitim süresi: {logreg_train_time:.4f} saniye")
print(f"Logistic Regression model test süresi: {logreg_test_time:.4f} saniye")

# Performans değerlendirmesi
print("Logistic Regression Model Performansı:")
print(classification_report(y_test, y_pred_logreg))
