import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Veri Yükleme 
data = pd.read_csv("C:\\Users\\pc\\OneDrive\\Masaüstü\\ders\\LAB\\LinearRegression\\housing.csv")
X = data[['total_rooms', 'median_income']]  # Özellikler
y = data['median_house_value']              # Hedef değişken

# 2. Bias terimini ekle
X.insert(0, 'Bias', 1)

# 3. Numpy array'e çevir
X_np = X.to_numpy()
y_np = y.to_numpy().reshape(-1, 1)

# 4. Least Squares çözümü
X_T = X_np.T
theta = np.linalg.inv(X_T @ X_np) @ X_T @ y_np

# 5. Tahmin yap
y_pred = X_np @ theta

# 6. Cost function (MSE)
cost = np.mean((y_np - y_pred) ** 2)
print("MSE (Least Squares):", cost)

# 7. Görselleştirme (median_income üzerinden)
plt.figure(figsize=(8, 5))
plt.scatter(X['median_income'], y, color='blue', label='Gerçek')
plt.scatter(X['median_income'], y_pred, color='red', label='Tahmin')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Linear Regression with Least Squares')
plt.legend()
plt.grid()
plt.show()

# 8. Theta değerleri
print("\nTheta (Katsayılar):")
for i, col in enumerate(X.columns):
    print(f"{col}: {theta[i][0]:.4f}")
